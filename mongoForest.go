package randomForest

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

type DataDTO[F Feature] struct {
	Input  []F     `bson:"input"`
	Reward float64 `bson:"reward"`
}

type MongoForest[F Feature] struct {
	*BaseForest[F]
	Labels   []float64
	Trees    []*RegressionTree[F]
	Range    float64
	Game     string
	database *mongo.Database `bson:"-"`
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

func NewMongoForest[F Feature](database *mongo.Database, treeCount int, samplesAmount, selectedFeatureAmount, featureCount int, r float64, game string) *MongoForest[F] {
	return &MongoForest[F]{
		Trees:    make([]*RegressionTree[F], 0),
		database: database,
		Range:    r,
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

func getData(collection mongo.Collection, count int) (*mongo.Cursor, error) {
	pipeline := mongo.Pipeline([]bson.D{{{Key: "$sample", Value: bson.D{{Key: "size", Value: count}}}}})
	return collection.Aggregate(context.Background(), pipeline)
}

func (forest *MongoForest[F]) BuildTree() *RegressionTree[F] {
	cursor, err := getData(*forest.database.Collection(fmt.Sprintf("steps_%s", forest.Game)), forest.NSize)
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
	tree.Root = buildRegressionNode(samples, samples_labels, forest.MFeatures, forest.MaxDepth)
	count := 0
	e := 0.0
	cursor, err = getData(*forest.database.Collection(fmt.Sprintf("steps_%s", forest.Game)), forest.NSize/10)
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

func (forest *MongoForest[F]) Predicate(input []F) float64 {
	total := 0.0
	for i := 0; i < len(forest.Trees); i++ {
		total += forest.Trees[i].Predicate(input)
	}
	avg := total / float64(len(forest.Trees))
	return avg
}

func (forest *MongoForest[F]) WeightedPredicate(input []F) float64 {
	total := 0.0
	v := 0.0
	for i := 0; i < len(forest.Trees); i++ {
		e := 1.0001 - forest.Trees[i].Validation
		w := 0.5 * math.Log(forest.Range*(1-e)/e)
		if w > 0 {
			v += forest.Trees[i].Predicate(input) * w
			total += w
		}
	}

	return v / total
}

func (forest *MongoForest[F]) DumpForest() {
	upsert := true
	_, err := forest.database.Collection("forests").ReplaceOne(context.Background(), bson.D{{Key: "game", Value: forest.Game}}, forest, &options.ReplaceOptions{Upsert: &upsert})
	if err != nil {
		panic(err)
	}
}

func LoadMongoForest[F Feature](database *mongo.Database, game string) *MongoForest[F] {
	result := database.Collection("forests").FindOne(context.Background(), bson.D{{Key: "game", Value: game}})
	if result.Err() != nil {
		panic(result.Err())
	}
	var forest MongoForest[F]
	result.Decode(&forest)
	forest.database = database
	return &forest
}
