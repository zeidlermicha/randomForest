package main

import (
	"fmt"

	"github.com/zeidlermicha/RF.go/RF/Regression"
	//"math"
)

func round(x float64) int {
	return int(x + 0.5)
}

func printTree[T Regression.Feature](tree *Regression.TreeNode[T], level int) {
	if tree == nil {
		return
	}
	fmt.Println(tree)
	if tree.Left != nil {
		for i := 0; i < level; i++ {
			fmt.Print("\t")
		}
		fmt.Print("left:")
		printTree(tree.Left, level+1)
	}
	if tree.Right != nil {
		for i := 0; i < level; i++ {
			fmt.Print("\t")
		}
		fmt.Print("rigth:")
		printTree(tree.Right, level+1)
	}
}

func main() {
	train_inputs := [][]int{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	train_targets := []float64{0, 1, 1, 0}

	forest := Regression.BuildForest(train_inputs, train_targets, 100, len(train_inputs), 2)

	for i := 0; i < len(train_inputs); i++ {
		x := train_inputs[i]
		fmt.Println(x)
		fmt.Println(round(forest.Predicate(x)))
	}

	printTree(forest.Trees[0].Root, 1)

}
