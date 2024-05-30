package randomForest

import (
	"math"
	"math/rand"
	"reflect"
)

type ColumnType int

const (
	CAT ColumnType = iota
	NUMERIC
)

type ClassificationTree[F Feature, L Label] struct {
	Root       *ClassificationNode[F, L]
	Validation float64
}

type ClassificationNode[F Feature, L Label] struct {
	Size    int
	Value   *F
	Left    *ClassificationNode[F, L]
	Right   *ClassificationNode[F, L]
	Column  int
	Labels  map[L]float64
	Measure float64
}

func getRandomRange(N int, M int) []int {
	return rand.Perm(N)[:M]
}

func getSamples[F Feature](ary [][]F, index []int) [][]F {
	result := make([][]F, len(index))
	for i := 0; i < len(index); i++ {
		result[i] = ary[index[i]]
	}
	return result
}

func getLabels[L any](ary []L, index []int) []L {
	result := make([]L, len(index))
	for i := 0; i < len(index); i++ {
		result[i] = ary[index[i]]
	}
	return result
}

func getEntropy[L Label](ep_map map[L]float64, total int) float64 {

	for k := range ep_map {
		ep_map[k] = ep_map[k] / float64(total) //normalize
	}

	entropy := 0.0
	for _, v := range ep_map {
		entropy += v * math.Log(1.0/v)
	}

	return entropy
}

func getBestGain[F Feature, L Label](samples [][]F, c int, samples_labels []L, column_type ColumnType, current_entropy float64) (float64, F, int, int) {
	var best_value F
	best_gain := 0.0
	best_total_r := 0
	best_total_l := 0

	uniq_values := make(map[F]int)
	for i := 0; i < len(samples); i++ {
		uniq_values[samples[i][c]] = 1
	}

	for value := range uniq_values {
		map_l := make(map[L]float64)
		map_r := make(map[L]float64)
		total_l := 0
		total_r := 0
		if column_type == CAT {
			for j := 0; j < len(samples); j++ {
				if samples[j][c] == value {
					total_l += 1
					map_l[samples_labels[j]] += 1.0
				} else {
					total_r += 1
					map_r[samples_labels[j]] += 1.0
				}
			}
		}
		if column_type == NUMERIC {
			for j := 0; j < len(samples); j++ {
				if samples[j][c] <= value {
					total_l += 1
					map_l[samples_labels[j]] += 1.0
				} else {
					total_r += 1
					map_r[samples_labels[j]] += 1.0
				}
			}
		}

		p1 := float64(total_r) / float64(len(samples))
		p2 := float64(total_l) / float64(len(samples))

		new_entropy := p1*getEntropy(map_r, total_r) + p2*getEntropy(map_l, total_l)
		//fmt.Println(new_entropy,current_entropy)
		entropy_gain := current_entropy - new_entropy

		if entropy_gain >= best_gain {
			best_gain = entropy_gain
			best_value = value
			best_total_l = total_l
			best_total_r = total_r
		}
	}

	return best_gain, best_value, best_total_l, best_total_r
}

func splitSamples[F Feature](samples [][]F, column_type ColumnType, c int, value F) (partL []int, partR []int) {
	partL = make([]int, 0, len(samples))
	partR = make([]int, 0, len(samples))
	if column_type == CAT {
		for j := 0; j < len(samples); j++ {
			if samples[j][c] == value {
				partL = append(partL, j)
			} else {
				partR = append(partR, j)
			}
		}
	}
	if column_type == NUMERIC {
		for j := 0; j < len(samples); j++ {
			if samples[j][c] <= value {
				partL = append(partL, j)
			} else {
				partR = append(partR, j)
			}
		}
	}

	return partL, partR
}

func buildNode[F Feature, L Label](samples [][]F, labels []L, selectedFeatureCount int, depth int) *ClassificationNode[F, L] {
	column_count := len(samples[0])
	//split_count := int(math.Log(float64(column_count)))
	split_count := selectedFeatureCount
	columns_choosen := getRandomRange(column_count, split_count)

	best_gain := 0.0

	var best_value F
	var best_column int
	var best_total_l int = 0
	var best_total_r int = 0
	var best_column_type ColumnType

	current_entropy_map := make(map[L]float64)
	for i := 0; i < len(labels); i++ {
		current_entropy_map[labels[i]] += 1
	}

	current_entropy := getEntropy(current_entropy_map, len(labels))

	for _, c := range columns_choosen {
		column_type := CAT

		if reflect.TypeOf(samples[0][c]) == reflect.TypeFor[float64]() {
			column_type = NUMERIC
		}

		gain, value, total_l, total_r := getBestGain(samples, c, labels, column_type, current_entropy)
		//fmt.Println("kkkkk",gain,part_l,part_r)
		if gain >= best_gain {
			best_gain = gain
			best_total_l = total_l
			best_total_r = total_r
			best_value = value
			best_column = c
			best_column_type = column_type
		}
	}

	if best_gain > 0 && best_total_l > 0 && best_total_r > 0 && depth > 0 {
		node := &ClassificationNode[F, L]{
			Size:    len(labels),
			Measure: current_entropy + best_gain,
		}
		node.Value = &best_value
		node.Column = best_column
		bestPartL, bestPartR := splitSamples(samples, best_column_type, best_column, best_value)
		node.Left = buildNode(getSamples(samples, bestPartL), getLabels(labels, bestPartL), selectedFeatureCount, depth-1)
		node.Right = buildNode(getSamples(samples, bestPartR), getLabels(labels, bestPartR), selectedFeatureCount, depth-1)
		return node
	}

	return genLeafNode[F, L](labels)
}

func genLeafNode[F Feature, L Label](labels []L) *ClassificationNode[F, L] {
	counter := make(map[L]int)
	e_map := make(map[L]float64)
	for _, v := range labels {
		counter[v] += 1
		e_map[v] += 1.0
	}

	node := &ClassificationNode[F, L]{
		Size:    len(labels),
		Measure: getEntropy(e_map, len(labels)),
	}
	node.Labels = make(map[L]float64)
	for l, v := range counter {
		node.Labels[l] = float64(v) / float64(len(labels))
	}
	//fmt.Println(node)
	return node
}

func (node *ClassificationNode[F, L]) predicate(input []F) map[L]float64 {
	if node.Value == nil { //leaf node
		return node.Labels
	}

	c := node.Column
	value := input[c]

	switch reflect.TypeOf(value) {
	case reflect.TypeFor[float64]():
		if value <= *node.Value && node.Left != nil {
			return node.Left.predicate(input)
		} else if node.Right != nil {
			return node.Right.predicate(input)
		}
	default:
		if value == *node.Value && node.Left != nil {
			return node.Left.predicate(input)
		} else if node.Right != nil {
			return node.Right.predicate(input)
		}
	}

	return nil
}

func (tree *ClassificationTree[F, L]) Predicate(input []F) map[L]float64 {
	return tree.Root.predicate(input)
}

func (tree ClassificationTree[F, L]) importance(nFeatures int) []float64 {
	imp := make([]float64, nFeatures)
	tree.Root.importance(imp)
	//normalize
	sum := 0.0
	for i := 0; i < nFeatures; i++ {
		sum += imp[i]
	}
	if sum > 0 {
		for i := 0; i < nFeatures; i++ {
			imp[i] = imp[i] / sum
		}
	}
	return imp
}

func (node ClassificationNode[F, L]) importance(imp []float64) {
	imp[node.Column] += float64(node.Size) * node.Measure
	if node.Left != nil {
		node.Left.importance(imp)
	}
	if node.Right != nil {
		node.Right.importance(imp)
	}
}
