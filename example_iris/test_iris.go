package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/zeidlermicha/randomForest"
)

func main() {

	start := time.Now()
	f, _ := os.Open("iris2.data")
	defer f.Close()
	content, _ := ioutil.ReadAll(f)
	s_content := string(content)
	lines := strings.Split(s_content, "\n")

	// set up variables for random forest
	inputs := make([][]float64, 0)
	targets := make([]string, 0)
	for _, line := range lines {

		line = strings.TrimRight(line, "\r\n")

		if len(line) == 0 {
			continue
		}
		tup := strings.Split(line, ",")
		pattern := tup[:len(tup)-1]
		target := tup[len(tup)-1]
		X := make([]float64, 0)
		for _, x := range pattern {
			f_x, _ := strconv.ParseFloat(x, 64)
			X = append(X, f_x)
		}
		inputs = append(inputs, X)

		targets = append(targets, target)
	}
	train_inputs := make([][]float64, 0)
	train_targets := make([]string, 0)

	test_inputs := make([][]float64, 0)
	test_targets := make([]string, 0)

	for i, x := range inputs {
		if i%3 == 0 {
			test_inputs = append(test_inputs, x)
		} else {
			train_inputs = append(train_inputs, x)
		}
	}

	for i, y := range targets {
		if i%3 == 0 {
			test_targets = append(test_targets, y)
		} else {
			train_targets = append(train_targets, y)
		}
	}

	forest := randomForest.NewClassificationForest[float64, string](10000, 100, 80, 4) //100 trees
	forest.Train(inputs, targets, 100)
	err_count := 0.0
	for i := 0; i < len(test_inputs); i++ {
		output := forest.WeightedPredicate(test_inputs[i])
		expect := test_targets[i]
		fmt.Println(output, expect)
		if output != expect {
			err_count += 1
		}
	}
	fmt.Println("success rate:", 1.0-err_count/float64(len(test_inputs)))

	fmt.Println(time.Since(start))

}
