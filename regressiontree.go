package randomForest

import (
	"math/rand"
	"reflect"
)

type RegressionTree[F Feature] struct {
	Root       *RegressionNode[F]
	Validation float64
}

type RegressionNode[F Feature] struct {
	Size    int
	Value   *F
	Left    *RegressionNode[F]
	Right   *RegressionNode[F]
	Column  int
	Label   float64
	Measure float64
}

func (tree RegressionTree[F]) importance(nFeatures int) []float64 {
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

func (node RegressionNode[F]) importance(imp []float64) {
	imp[node.Column] += float64(node.Size) * node.Measure
	if node.Left != nil {
		node.Left.importance(imp)
	}
	if node.Right != nil {
		node.Right.importance(imp)
	}
}

func getMSE(labels []float64) float64 {
	if len(labels) == 0 {
		return 0.0
	}
	total := 0.0
	for _, x := range labels {
		total += x
	}
	avg := total / float64(len(labels))
	mse := 0.0
	for _, x := range labels {
		delta := x - avg
		mse += delta * delta
	}
	mse = mse / float64(len(labels))
	return mse
}

func getBestMSEGain[F Feature](samples [][]F, c int, samples_labels []float64, column_type ColumnType, current_mse float64) (float64, F, int, int) {
	var best_value F
	best_gain := 0.0
	best_total_r := 0
	best_total_l := 0

	uniq_values := make(map[F]int)
	for i := 0; i < len(samples); i++ {
		uniq_values[samples[i][c]] = 1
	}

	for value := range uniq_values {
		labels_l := make([]float64, 0)
		labels_r := make([]float64, 0)
		total_l := 0
		total_r := 0
		if column_type == CAT {
			for j := 0; j < len(samples); j++ {
				if samples[j][c] == value {
					total_l += 1
					labels_l = append(labels_l, samples_labels[j])
				} else {
					total_r += 1
					labels_r = append(labels_r, samples_labels[j])
				}
			}
		}
		if column_type == NUMERIC {
			for j := 0; j < len(samples); j++ {
				if samples[j][c] <= value {
					total_l += 1
					labels_l = append(labels_l, samples_labels[j])
				} else {
					total_r += 1
					labels_r = append(labels_r, samples_labels[j])
				}
			}
		}

		p1 := float64(total_r) / float64(len(samples))
		p2 := float64(total_l) / float64(len(samples))

		new_mse := p1*getMSE(labels_r) + p2*getMSE(labels_l)

		//fmt.Println(new_mse,part_l,part_r)
		mse_gain := current_mse - new_mse

		if mse_gain >= best_gain {
			best_gain = mse_gain
			best_value = value
			best_total_l = total_l
			best_total_r = total_r
		}
	}

	return best_gain, best_value, best_total_l, best_total_r
}

func buildRegressionNode[F Feature](samples [][]F, samples_labels []float64, selected_feature_count int) *RegressionNode[F] {

	column_count := len(samples[0])
	//split_count := int(math.Log(float64(column_count)))
	split_count := selected_feature_count
	columns_choosen := getRandomRange(column_count, split_count)

	best_gain := 0.0
	var best_value F
	var best_column int
	var best_total_l int = 0
	var best_total_r int = 0
	var best_column_type ColumnType

	current_mse := getMSE(samples_labels)

	for _, c := range columns_choosen {
		column_type := CAT

		if reflect.TypeOf(samples[0][c]) == reflect.TypeFor[float64]() {
			column_type = NUMERIC
		}

		gain, value, total_l, total_r := getBestMSEGain(samples, c, samples_labels, column_type, current_mse)
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

	if best_gain > 0 && best_total_l > 0 && best_total_r > 0 {
		//fmt.Println(best_part_l,best_part_r)
		node := &RegressionNode[F]{
			Size:    len(samples_labels),
			Measure: current_mse + best_gain,
		}
		node.Value = &best_value
		node.Column = best_column
		bestPartL, bestPartR := splitSamples(samples, best_column_type, best_column, best_value)
		node.Left = buildRegressionNode(getSamples(samples, bestPartL), getLabels(samples_labels, bestPartL), selected_feature_count)
		node.Right = buildRegressionNode(getSamples(samples, bestPartR), getLabels(samples_labels, bestPartR), selected_feature_count)
		return node
	}

	return genRegressionLeafNode[F](samples_labels)

}

func genRegressionLeafNode[F Feature](labels []float64) *RegressionNode[F] {
	total := 0.0
	for _, x := range labels {
		total += x
	}
	avg := total / float64(len(labels))
	node := &RegressionNode[F]{
		Size:    len(labels),
		Measure: getMSE(labels),
	}
	node.Label = avg
	//fmt.Println(node)
	return node
}

func predicate[T Feature](node *RegressionNode[T], input []T) float64 {

	if reflect.ValueOf(node.Value).IsNil() { //leaf node
		return node.Label
	}

	c := node.Column
	value := input[c]

	switch reflect.TypeOf(value) {
	case reflect.TypeFor[float64]():
		if value <= *node.Value && node.Left != nil {
			return predicate(node.Left, input)
		} else if node.Right != nil {
			return predicate(node.Right, input)
		}
	default:
		if value == *node.Value && node.Left != nil {
			return predicate(node.Left, input)
		} else if node.Right != nil {
			return predicate(node.Right, input)
		}
	}

	return 0
}

func BuildTree[F Feature](inputs [][]F, labels []float64, samples_count, selected_feature_count int) *RegressionTree[F] {

	samples := make([][]F, samples_count)
	samples_labels := make([]float64, samples_count)
	for i := 0; i < samples_count; i++ {
		j := int(rand.Float64() * float64(len(inputs)))
		samples[i] = inputs[j]
		samples_labels[i] = labels[j]
	}

	tree := &RegressionTree[F]{}
	tree.Root = buildRegressionNode(samples, samples_labels, selected_feature_count)

	return tree
}

func (tree *RegressionTree[F]) Predicate(input []F) float64 {
	return predicate(tree.Root, input)
}
