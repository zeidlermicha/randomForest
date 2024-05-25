package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"

	"github.com/zeidlermicha/randomForest"
)

func ReadMNISTLabels(r io.Reader) (labels []byte) {
	header := [2]int32{}
	binary.Read(r, binary.BigEndian, &header)
	labels = make([]byte, header[1])
	r.Read(labels)
	return
}

func ReadMNISTImages(r io.Reader) (images [][]byte, width, height int) {
	header := [4]int32{}
	binary.Read(r, binary.BigEndian, &header)
	images = make([][]byte, header[1])
	width, height = int(header[2]), int(header[3])
	for i := 0; i < len(images); i++ {
		images[i] = make([]byte, width*height)
		r.Read(images[i])
	}
	return
}

func ImageString(buffer []byte, height, width int) (out string) {
	for i, y := 0, 0; y < height; y++ {
		for x := 0; x < width; x++ {
			if buffer[i] > 128 {
				out += "#"
			} else {
				out += " "
			}
			i++
		}
		out += "\n"
	}
	return
}

func OpenFile(path string) *os.File {
	file, err := os.Open(path)
	if err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}
	return file
}

func prepareX(M [][]byte) [][]float64 {
	rows := len(M)
	result := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, len(M[i]))
		for j := 0; j < len(M[i]); j++ {
			result[i][j] = float64(M[i][j])
		}
	}
	return result
}

func prepareY(N []byte) []string {
	result := make([]string, len(N))
	for i := 0; i < len(result); i++ {
		result[i] = fmt.Sprintf("%d", int(N[i]))
	}
	return result
}

func main() {
	runtime.GOMAXPROCS(8)
	sourceLabelFile := flag.String("sl", "", "source label file")
	sourceImageFile := flag.String("si", "", "source image file")
	testLabelFile := flag.String("tl", "", "test label file")
	testImageFile := flag.String("ti", "", "test image file")

	flag.Parse()

	if *sourceLabelFile == "" || *sourceImageFile == "" {
		flag.Usage()
		os.Exit(-2)
	}

	fmt.Println("Loading training data...")
	labelData := ReadMNISTLabels(OpenFile(*sourceLabelFile))
	imageData, width, height := ReadMNISTImages(OpenFile(*sourceImageFile))

	fmt.Println(len(imageData), len(imageData[0]), width, height)
	fmt.Println(len(labelData), labelData[0:10])

	inputs := prepareX(imageData)
	targets := prepareY(labelData)

	//fmt.Println(inputs[0],targets[0])

	//forest := RF.BuildForest(inputs, targets, 100, 2000, 30) //100 tries, 2000 samples, 30 features
	forest := randomForest.NewClassificationForest[float64, string](1000, 100, 2000, 30)
	forest.Train(inputs, targets, 100)
	//RF.DumpForest(forest,"rf.bin")
	var testLabelData []byte
	var testImageData [][]byte
	if *testLabelFile != "" && *testImageFile != "" {
		fmt.Println("Loading test data...")
		testLabelData = ReadMNISTLabels(OpenFile(*testLabelFile))
		testImageData, _, _ = ReadMNISTImages(OpenFile(*testImageFile))
	}

	test_inputs := prepareX(testImageData)
	test_targets := prepareY(testLabelData)

	correct_ct := 0
	for i, p := range test_inputs {
		y := forest.Predicate(p)
		yy := test_targets[i]
		//fmt.Println(y,yy)
		if y == yy {
			correct_ct += 1
		}
	}

	fmt.Println("correct rate: ", float64(correct_ct)/float64(len(test_inputs)), correct_ct, len(test_inputs))
}
