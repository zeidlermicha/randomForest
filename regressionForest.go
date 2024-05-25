package randomForest

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
	"time"
)

type RegressionForest[F Feature] struct {
	*BaseForest[F]
	Labels []float64
	Trees  []*RegressionTree[F]
	Range  float64
}

func (f RegressionForest[F]) Importance() []float64 {
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

func NewRegressionForest[F Feature](bufferSize int, treeLimit int, samplesAmount, selectedFeatureAmount float64) *RegressionForest[F] {
	return &RegressionForest[F]{
		Trees:      make([]*RegressionTree[F], 0),
		BaseForest: &BaseForest[F]{BufferSize: bufferSize, TreeLimit: treeLimit, NSizeFactor: samplesAmount, MFeaturesFactor: selectedFeatureAmount},
	}
}

func (forest *RegressionForest[F]) Train(inputs [][]F, labels []float64, treesAmount int) {
	forest.Data = append(forest.Data, inputs...)
	if len(forest.Data) > forest.BufferSize {
		forest.Data = shiftLeft(forest.Data, len(forest.Data)-forest.BufferSize)
	}
	forest.Labels = append(forest.Labels, labels...)
	if len(forest.Labels) > forest.BufferSize {
		forest.Labels = shiftLeft(forest.Labels, len(forest.Labels)-forest.BufferSize)
	}
	vMin := math.MaxFloat64
	vMax := -math.MaxFloat64
	for _, v := range forest.Labels {
		if v < vMin {
			vMin = v
		}
		if v > vMax {
			vMax = v
		}
	}

	forest.Range = vMax - vMin

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

func (forest *RegressionForest[F]) BuildTree() *RegressionTree[F] {
	samples := make([][]F, forest.NSize)
	samples_labels := make([]float64, forest.NSize)
	used := make([]bool, len(forest.Data))
	for i := 0; i < forest.NSize; i++ {
		j := rand.Intn(len(forest.Data))
		samples[i] = forest.Data[j]
		samples_labels[i] = forest.Labels[j]
		used[i] = true
	}

	tree := &RegressionTree[F]{}
	tree.Root = buildRegressionNode(samples, samples_labels, forest.MFeatures)
	count := 0
	e := 0.0
	for i := 0; i < len(forest.Data); i++ {
		if !used[i] {
			count++
			e += tree.Predicate(forest.Data[i])
		}
	}
	tree.Validation = math.Abs(e / float64(count))
	return tree
}

func (forest *RegressionForest[F]) Predicate(input []F) float64 {
	total := 0.0
	for i := 0; i < len(forest.Trees); i++ {
		total += forest.Trees[i].Predicate(input)
	}
	avg := total / float64(len(forest.Trees))
	return avg
}

func (forest *RegressionForest[F]) WeightedPredicate(input []F) float64 {
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

func (forest *RegressionForest[F]) DumpForest(fileName string) {
	out_f, err := os.OpenFile(fileName, os.O_CREATE|os.O_RDWR, 0777)
	if err != nil {
		panic("failed to create " + fileName)
	}
	defer out_f.Close()
	encoder := json.NewEncoder(out_f)
	encoder.Encode(forest)
}

func LoadRegressionForest[F Feature](fileName string) *RegressionForest[F] {
	in_f, err := os.Open(fileName)
	if err != nil {
		panic("failed to open " + fileName)
	}
	defer in_f.Close()
	decoder := json.NewDecoder(in_f)
	forest := &RegressionForest[F]{}
	decoder.Decode(forest)
	return forest
}
