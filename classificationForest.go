package randomForest

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sync"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

var NUM_CPU = runtime.NumCPU()

func shiftLeft[E any](nums []E, n int) []E {
	n = n % len(nums)
	return nums[n:len(nums):len(nums)]
}

type BuildTreeFunc[F Feature, L Label] func(inputs [][]F, labels []L, samplesAmount, selectedFeatureAmount int) *ClassificationTree[F, L]

type Feature interface {
	~string | ~float64 | ~int
}

type Label interface {
	~string | ~int
}

type ClassificationDTO[F Feature, L Label] struct {
	Input []F `bson:"input"`
	Label L   `bson:"label"`
}

type BaseForest[F Feature] struct {
	Features        int
	MFeatures       int
	MFeaturesFactor float64
	NSize           int
	NSizeFactor     float64
	Data            [][]F `json:"-"`
	BufferSize      int
	TreeLimit       int
	MaxDepth        int
}

type ClassificationForest[F Feature, L Label] struct {
	*BaseForest[F]
	Trees   []*ClassificationTree[F, L]
	Labels  []L `json:"-"`
	Classes int
}

type MongoClassForest[F Feature, L Label] struct {
	*BaseForest[F]
	Trees    []*ClassificationTree[F, L]
	Labels   []L `json:"-"`
	Classes  int
	Game     string
	database *mongo.Database `json:"-"`
}

func maxLabel[L Label](votes map[L]float64) (L, float64) {
	var maxLabel L
	maxValue := 0.0
	for l, v := range votes {
		if v > maxValue {
			maxValue = v
			maxLabel = l
		}
	}

	return maxLabel, maxValue
}

func (f ClassificationForest[T, L]) Importance() []float64 {
	imp := make([]float64, f.Features)
	for i := 0; i < len(f.Trees); i++ {
		z := f.Trees[i].importance(f.Features)
		for i := 0; i < f.Features; i++ {
			imp[i] += z[i]
		}
	}
	for i := 0; i < f.Features; i++ {
		imp[i] = imp[i] / float64(len(f.Trees))
	}
	return imp
}

func (f MongoClassForest[T, L]) Importance() []float64 {
	imp := make([]float64, f.Features)
	for i := 0; i < len(f.Trees); i++ {
		z := f.Trees[i].importance(f.Features)
		for i := 0; i < f.Features; i++ {
			imp[i] += z[i]
		}
	}
	for i := 0; i < f.Features; i++ {
		imp[i] = imp[i] / float64(len(f.Trees))
	}
	return imp
}

func NewClassificationForest[F Feature, L Label](bufferSize int, treeLimit int, samplesAmount, selectedFeatureAmount float64) *ClassificationForest[F, L] {
	return &ClassificationForest[F, L]{
		Trees:      make([]*ClassificationTree[F, L], 0),
		BaseForest: &BaseForest[F]{BufferSize: bufferSize, TreeLimit: treeLimit, NSizeFactor: samplesAmount, MFeaturesFactor: selectedFeatureAmount},
	}
}

func NewMongoClassForest[F Feature, L Label](database *mongo.Database, treeCount int, samplesAmount, selectedFeatureAmount, featureCount int, classes int, game string) *MongoClassForest[F, L] {
	return &MongoClassForest[F, L]{
		Trees:    make([]*ClassificationTree[F, L], 0),
		database: database,
		Classes:  classes,
		Game:     game,
		BaseForest: &BaseForest[F]{
			TreeLimit: treeCount,
			NSize:     samplesAmount,
			MFeatures: selectedFeatureAmount,
			Features:  featureCount,
			MaxDepth:  10,
		},
	}
}

func (forest *ClassificationForest[F, L]) Train(inputs [][]F, labels []L, treesAmount int) {
	forest.Data = append(forest.Data, inputs...)
	if len(forest.Data) > forest.BufferSize {
		forest.Data = shiftLeft(forest.Data, len(forest.Data)-forest.BufferSize)
	}
	forest.Labels = append(forest.Labels, labels...)
	if len(forest.Labels) > forest.BufferSize {
		forest.Labels = shiftLeft(forest.Labels, len(forest.Labels)-forest.BufferSize)
	}
	classMap := make(map[L]bool)
	for _, c := range forest.Labels {
		if _, ok := classMap[c]; !ok {
			forest.Classes += 1
			classMap[c] = true
		}
	}

	forest.Features = len(inputs[0])
	forest.MFeatures = int(float64(forest.Features) * forest.MFeaturesFactor)
	forest.NSize = int(float64(len(forest.Data)) * forest.NSizeFactor)
	prog_counter := 0
	mutex := &sync.Mutex{}
	s := make(chan bool, NUM_CPU)
	for i := 0; i < treesAmount; i++ {
		s <- true
		go func(x int) {
			defer func() { <-s }()

			fmt.Printf(">> %v buiding %vth tree...\n", time.Now(), x)
			tree := forest.BuildTree()
			forest.Trees = append(forest.Trees, tree)
			//fmt.Printf("<< %v the %vth tree is done.\n",time.Now(), x)
			mutex.Lock()
			prog_counter += 1
			fmt.Printf("%v tranning progress %.0f%%\n", time.Now(), float64(prog_counter)/float64(treesAmount)*100)
			mutex.Unlock()
		}(i)
	}

	for i := 0; i < NUM_CPU; i++ {
		s <- true
	}
}

func (forest *MongoClassForest[F, L]) Train() {
	prog_counter := 0
	mutex := &sync.Mutex{}
	s := make(chan bool, NUM_CPU)
	for i := 0; i < forest.TreeLimit; i++ {
		s <- true
		go func(x int) {
			defer func() { <-s }()

			fmt.Printf(">> %v buiding %vth tree...\n", time.Now(), x)
			tree := forest.BuildTree()
			forest.Trees = append(forest.Trees, tree)
			//fmt.Printf("<< %v the %vth tree is done.\n",time.Now(), x)
			mutex.Lock()
			prog_counter += 1
			fmt.Printf("%v tranning progress %.0f%%\n", time.Now(), float64(prog_counter)/float64(forest.TreeLimit)*100)
			mutex.Unlock()
		}(i)
	}

	for i := 0; i < NUM_CPU; i++ {
		s <- true
	}
}

func (forest *ClassificationForest[F, L]) BuildTree() *ClassificationTree[F, L] {
	samples := make([][]F, forest.NSize)
	samples_labels := make([]L, forest.NSize)
	used := make([]bool, len(forest.Data))
	for i := 0; i < forest.NSize; i++ {
		j := rand.Intn(len(forest.Data))
		samples[i] = forest.Data[j]
		samples_labels[i] = forest.Labels[j]
		used[i] = true
	}

	tree := &ClassificationTree[F, L]{}
	tree.Root = buildNode(samples, samples_labels, forest.MFeatures, forest.MaxDepth)
	count := 0
	e := 0.0
	for i := 0; i < len(forest.Data); i++ {
		if !used[i] {
			count++
			v := tree.Predicate(forest.Data[i])
			e += v[forest.Labels[i]]
		}
	}
	tree.Validation = e / float64(count)
	return tree
}

func (forest *MongoClassForest[F, L]) BuildTree() *ClassificationTree[F, L] {

	cursor, err := getData(*forest.database.Collection(fmt.Sprintf("steps_%s", forest.Game)), forest.NSize)
	if err != nil {
		panic(err)
	}
	samples := make([][]F, 0, forest.NSize)
	samples_labels := make([]L, 0, forest.NSize)
	for cursor.Next(context.Background()) {
		var data ClassificationDTO[F, L]
		err := cursor.Decode(&data)
		if err != nil {
			panic(err)
		}
		samples = append(samples, data.Input[:len(data.Input)-1])
		samples_labels = append(samples_labels, data.Label)
	}
	tree := &ClassificationTree[F, L]{}
	tree.Root = buildNode(samples, samples_labels, forest.MFeatures, forest.MaxDepth)
	count := 0
	e := 0.0
	cursor, err = getData(*forest.database.Collection(fmt.Sprintf("steps_%s", forest.Game)), forest.NSize/10)
	if err != nil {
		panic(err)
	}
	for cursor.Next(context.Background()) {
		var data ClassificationDTO[F, L]
		cursor.Decode(&data)
		count++
		v := tree.Predicate(data.Input[:len(data.Input)-1])
		e += v[data.Label]
	}
	tree.Validation = math.Abs(e / float64(count))
	return tree
}

func (self *ClassificationForest[F, L]) Predicate(input []F) L {
	l, _ := maxLabel(self.PredicateWithData(input))
	return l
}

func (forest *MongoClassForest[F, L]) Predicate(input []F) L {
	l, _ := maxLabel((forest.PredicateWithData(input)))
	return l
}

func (self *ClassificationForest[F, L]) PredicateWithData(input []F) map[L]float64 {
	counter := make(map[L]float64)
	for i := 0; i < len(self.Trees); i++ {
		tree_counter := self.Trees[i].Predicate(input)
		total := 0.0
		for _, v := range tree_counter {
			total += float64(v)
		}
		for k, v := range tree_counter {
			counter[k] += float64(v) / total
		}
	}

	return counter

}

func (self *MongoClassForest[F, L]) PredicateWithData(input []F) map[L]float64 {
	counter := make(map[L]float64)
	for i := 0; i < len(self.Trees); i++ {
		tree_counter := self.Trees[i].Predicate(input)
		total := 0.0
		for _, v := range tree_counter {
			total += float64(v)
		}
		for k, v := range tree_counter {
			counter[k] += float64(v) / total
		}
	}

	return counter

}

func (forest *ClassificationForest[F, L]) WeightedPredicate(input []F) L {
	l, _ := maxLabel(forest.WeightedPredicateWithData(input))
	return l
}

func (forest *ClassificationForest[F, L]) WeightedPredicateWithData(input []F) map[L]float64 {
	counter := make(map[L]float64)
	total := 0.0
	for i := 0; i < len(forest.Trees); i++ {
		e := 1.0001 - forest.Trees[i].Validation
		w := 0.5 * math.Log(float64(forest.Classes-1)*(1-e)/e)
		if w > 0 {
			tree_counter := forest.Trees[i].Predicate(input)
			for label, v := range tree_counter {
				counter[label] += float64(v) * w
			}
			total += w
		}
	}
	for label, v := range counter {
		counter[label] = v / total
	}
	return counter
}

func (forest *ClassificationForest[F, L]) DumpForest(fileName string) {
	out_f, err := os.OpenFile(fileName, os.O_CREATE|os.O_RDWR, 0777)
	if err != nil {
		panic("failed to create " + fileName)
	}
	defer out_f.Close()
	encoder := json.NewEncoder(out_f)
	encoder.Encode(forest)
}

func LoadForest[T Feature, L Label](fileName string) *ClassificationForest[T, L] {
	in_f, err := os.Open(fileName)
	if err != nil {
		panic("failed to open " + fileName)
	}
	defer in_f.Close()
	decoder := json.NewDecoder(in_f)
	forest := &ClassificationForest[T, L]{}
	decoder.Decode(forest)
	return forest
}

func (forest *MongoClassForest[F, L]) DumpForest() {
	upsert := true
	_, err := forest.database.Collection("class_forests").ReplaceOne(context.Background(), bson.D{{Key: "game", Value: forest.Game}}, forest, &options.ReplaceOptions{Upsert: &upsert})
	if err != nil {
		panic(err)
	}
}

func LoadMongoClassForest[F Feature, L Label](database *mongo.Database, game string) *MongoClassForest[F, L] {
	result := database.Collection("class_forests").FindOne(context.Background(), bson.D{{Key: "game", Value: game}})
	if result.Err() != nil {
		panic(result.Err())
	}
	var forest MongoClassForest[F, L]
	result.Decode(&forest)
	forest.database = database
	return &forest
}
