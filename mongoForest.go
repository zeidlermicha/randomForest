package randomForest

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sync"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
)

type DataDTO[F Feature] struct {
	Input  []F     `bson:"input"`
	Reward float64 `bson:"reward"`
}

type MongoForest[F Feature] struct {
	*BaseForest[F]
	Labels     []float64
	Trees      []*RegressionTree[F]
	Range      float64
	Game       string
	Collection *mongo.Collection `bson:"-"`
}

func (f MongoForest[F]) Importance() []float64 {
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

func NewMongoForest[F Feature](collection *mongo.Collection, treeCount int, samplesAmount, selectedFeatureAmount, featureCount int, r float64, game string) *MongoForest[F] {
	return &MongoForest[F]{
		Trees:      make([]*RegressionTree[F], 0),
		Collection: collection,
		Range:      r,
		Game:       game,
		BaseForest: &BaseForest[F]{TreeLimit: treeCount, NSize: samplesAmount, MFeatures: selectedFeatureAmount, Features: featureCount},
	}
}

func (forest *MongoForest[F]) Train() {
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

func getData(collection mongo.Collection, game string, count int) (*mongo.Cursor, error) {
	pipeline := mongo.Pipeline([]bson.D{{{Key: "$match", Value: bson.D{{Key: "game", Value: game}}}}, {{Key: "$sample", Value: bson.D{{Key: "size", Value: count}}}}})
	return collection.Aggregate(context.Background(), pipeline)
}

func (forest *MongoForest[F]) BuildTree() *RegressionTree[F] {
	cursor, err := getData(*forest.Collection, forest.Game, forest.NSize)
	if err != nil {
		panic(err)
	}
	samples := make([][]F, 0, forest.NSize)
	samples_labels := make([]float64, 0, forest.NSize)
	for cursor.Next(context.Background()) {
		var data DataDTO[F]
		err := cursor.Decode(&data)
		if err != nil {
			panic(err)
		}
		samples = append(samples, data.Input)
		samples_labels = append(samples_labels, data.Reward)
	}

	tree := &RegressionTree[F]{}
	tree.Root = buildRegressionNode(samples, samples_labels, forest.MFeatures)
	count := 0
	e := 0.0
	cursor, err = getData(*forest.Collection, forest.Game, forest.NSize/10)
	if err != nil {
		panic(err)
	}
	for cursor.Next(context.Background()) {
		var data DataDTO[F]
		cursor.Decode(&data)
		count++
		e += tree.Predicate(data.Input)
	}
	tree.Validation = math.Abs(e / float64(count))
	return tree
}

func (forest *MongoForest[F]) DumpForest(fileName string) {
	out_f, err := os.OpenFile(fileName, os.O_CREATE|os.O_RDWR, 0777)
	if err != nil {
		panic("failed to create " + fileName)
	}
	defer out_f.Close()
	encoder := json.NewEncoder(out_f)
	encoder.Encode(forest)
}

func LoadMongoForest[F Feature](fileName string) *MongoForest[F] {
	in_f, err := os.Open(fileName)
	if err != nil {
		panic("failed to open " + fileName)
	}
	defer in_f.Close()
	decoder := json.NewDecoder(in_f)
	forest := &MongoForest[F]{}
	decoder.Decode(forest)
	return forest
}
