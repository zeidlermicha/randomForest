package main

import (
	"fmt"

	"github.com/zeidlermicha/randomForest"
	//"math"
)

func round(x float64) int {
	return int(x + 0.5)
}

func main() {
	train_inputs := [][]int{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	train_targets := []float64{0, 1, 1, 0}

	//	forest := Regression.BuildForest(train_inputs, train_targets, 100, 4, 2)
	forest := randomForest.NewRegressionForest[int](10000, 100, 4, 2)
	forest.Train(train_inputs, train_targets, 100)
	for i := 0; i < len(train_inputs); i++ {
		x := train_inputs[i]
		fmt.Println(x)
		fmt.Println(round(forest.Predicate(x)))
	}

}
