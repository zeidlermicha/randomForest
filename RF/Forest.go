package RF

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sync"
	"time"
)

type Feature interface {
	string | float64
}

type Forest[T Feature] struct {
	Trees []*Tree[T]
}

func BuildForest[T Feature](inputs [][]T, labels []string, treesAmount, samplesAmount, selectedFeatureAmount int) *Forest[T] {
	forest := &Forest[T]{}
	forest.Trees = make([]*Tree[T], treesAmount)
	prog_counter := 0
	mutex := &sync.Mutex{}
	var wait sync.WaitGroup
	for i := 0; i < treesAmount; i++ {
		wait.Add(1)
		go func(x int) {
			fmt.Printf(">> %v buiding %vth tree...\n", time.Now(), x)
			forest.Trees[x] = BuildTree(inputs, labels, samplesAmount, selectedFeatureAmount)
			//fmt.Printf("<< %v the %vth tree is done.\n",time.Now(), x)
			mutex.Lock()
			prog_counter += 1
			fmt.Printf("%v tranning progress %.0f%%\n", time.Now(), float64(prog_counter)/float64(treesAmount)*100)
			mutex.Unlock()
			wait.Done()
		}(i)
	}

	wait.Wait()

	fmt.Println("all done.")
	return forest
}

func DefaultForest[T Feature](inputs [][]T, labels []string, treesAmount int) *Forest[T] {
	m := int(math.Sqrt(float64(len(inputs[0]))))
	n := int(math.Sqrt(float64(len(inputs))))
	return BuildForest(inputs, labels, treesAmount, n, m)
}

func (self *Forest[T]) Predicate(input []T) string {
	counter := make(map[string]float64)
	for i := 0; i < len(self.Trees); i++ {
		tree_counter := PredicateTree(self.Trees[i], input)
		total := 0.0
		for _, v := range tree_counter {
			total += float64(v)
		}
		for k, v := range tree_counter {
			counter[k] += float64(v) / total
		}
	}

	max_c := 0.0
	max_label := ""
	for k, v := range counter {
		if v >= max_c {
			max_c = v
			max_label = k
		}
	}
	return max_label
}

func DumpForest[T Feature](forest *Forest[T], fileName string) {
	out_f, err := os.OpenFile(fileName, os.O_CREATE|os.O_RDWR, 0777)
	if err != nil {
		panic("failed to create " + fileName)
	}
	defer out_f.Close()
	encoder := json.NewEncoder(out_f)
	encoder.Encode(forest)
}

func LoadForest[T Feature](fileName string) *Forest[T] {
	in_f, err := os.Open(fileName)
	if err != nil {
		panic("failed to open " + fileName)
	}
	defer in_f.Close()
	decoder := json.NewDecoder(in_f)
	forest := &Forest[T]{}
	decoder.Decode(forest)
	return forest
}
