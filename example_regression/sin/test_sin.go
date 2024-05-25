package main

import (
	"fmt"
	"math"
	"os"

	"github.com/zeidlermicha/randomForest"
)

func main() {
	out_f, _ := os.OpenFile("sin.out", os.O_CREATE|os.O_RDWR, 0777)
	defer out_f.Close()

	train_inputs := make([][]float64, 100)
	train_targets := make([]float64, 100)

	for i := 0; i < len(train_inputs); i++ {
		train_inputs[i] = []float64{float64(i) / 20.0}
		train_targets[i] = math.Sin(train_inputs[i][0])
	}

	//forest := Regression.BuildForest(train_inputs, train_targets, 100, len(train_inputs), 1)
	forest := randomForest.NewRegressionForest[float64](10000, 100, 80, 1)
	forest.Train(train_inputs, train_targets, 100)
	total := 0.0
	totalW := 0.0
	count := 0.0
	for i := 0; i < 2000; i++ {
		count += 1
		x := []float64{float64(i) / 400.0}
		p := forest.Predicate(x)
		pw := forest.WeightedPredicate(x)
		fmt.Fprintln(out_f, x[0], p, pw)
		total = math.Abs(math.Sin(x[0])-p) * math.Abs(math.Sin(x[0]-p))
		totalW = math.Abs(math.Sin(x[0])-pw) * math.Abs(math.Sin(x[0])-pw)
	}

	fmt.Println(math.Sqrt(total / count))
	fmt.Println(math.Sqrt(totalW / count))

}
