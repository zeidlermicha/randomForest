package main

import (
	"fmt"

	"github.com/zeidlermicha/randomForest"
	//"math"
)

func Q(L []int) []float64 {
	tmp := make([]float64, 10)
	for i := 0; i < len(tmp); i++ {
		tmp[i] = 0.0
	}
	for _, x := range L {
		tmp[x] = tmp[x] + 1.0
	}
	return tmp
}

func R(F int) float64 {
	return float64(F)
}

func round(x float64) int {
	return int(x + 0.5)
}

func main() {

	inputs := make([][]float64, 0)
	inputs = append(inputs, Q([]int{7, 1, 1, 1}))
	inputs = append(inputs, Q([]int{8, 8, 0, 9}))
	inputs = append(inputs, Q([]int{2, 1, 7, 2}))
	inputs = append(inputs, Q([]int{6, 6, 6, 6}))
	inputs = append(inputs, Q([]int{1, 1, 1, 1}))
	inputs = append(inputs, Q([]int{2, 2, 2, 2}))
	inputs = append(inputs, Q([]int{7, 6, 6, 2}))
	inputs = append(inputs, Q([]int{9, 3, 1, 3}))
	inputs = append(inputs, Q([]int{0, 0, 0, 0}))
	inputs = append(inputs, Q([]int{5, 5, 5, 5}))
	inputs = append(inputs, Q([]int{8, 1, 9, 3}))
	inputs = append(inputs, Q([]int{8, 0, 9, 6}))
	inputs = append(inputs, Q([]int{4, 3, 9, 8}))
	inputs = append(inputs, Q([]int{9, 4, 7, 5}))
	inputs = append(inputs, Q([]int{9, 0, 3, 8}))
	inputs = append(inputs, Q([]int{3, 1, 4, 8}))

	//R(0),R(6),R(0),R(4),R(0),R(0),R(2),R(1),R(4),R(0),R(3),R(5),R(3),R(1),R(4),R(2)
	targets := make([]float64, 0)
	targets = append(targets, R(0))
	targets = append(targets, R(6))
	targets = append(targets, R(0))
	targets = append(targets, R(4))
	targets = append(targets, R(0))
	targets = append(targets, R(0))
	targets = append(targets, R(2))
	targets = append(targets, R(1))
	targets = append(targets, R(4))
	targets = append(targets, R(0))
	targets = append(targets, R(3))
	targets = append(targets, R(5))
	targets = append(targets, R(3))
	targets = append(targets, R(1))
	targets = append(targets, R(4))
	targets = append(targets, R(2))

	fmt.Println(inputs)
	fmt.Println(targets)

	//forest := (inputs, targets, 100, len(inputs), 10)
	forest := randomForest.NewRegressionForest[float64](10000, 100, 14, 10)
	forest.Train(inputs, targets, 100)
	inputs = append(inputs, Q([]int{0, 2, 4, 8}))
	inputs = append(inputs, Q([]int{2, 5, 8, 1}))

	targets = append(targets, R(3))
	targets = append(targets, R(2))

	for i, p := range inputs {
		fmt.Println(targets[i], round(forest.Predicate(p)))
	}

	fmt.Println("")
}
