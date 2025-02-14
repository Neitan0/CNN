import MLX
import MLXOptimizers
import MLXLinalg
import MLXNN
import MLXRandom
import Foundation
import CoreGraphics
import ImageIO


func loadDataset(quadradosPath: String, naoQuadradosPath: String) -> (images: [(MLXArray)], labels: [Int])? {
    // Função auxiliar para carregar imagens de uma pasta
    func loadImages(from path: String, label: Int) -> ([(MLXArray)], [Int])? {
        guard let datasetURL = Bundle.module.url(forResource: path, withExtension: nil) else {
            print("Erro ao encontrar o diretório: \(path)")
            return nil
        }
        
        do {
            let fileURLs = try FileManager.default.contentsOfDirectory(at: datasetURL, includingPropertiesForKeys: nil, options: [])
            let imageURLs = fileURLs.filter { $0.pathExtension.lowercased() == "png" || $0.pathExtension.lowercased() == "jpg" }
            var mlxArrays: [(MLXArray)] = []
            var labels: [Int] = []
            
            for imageURL in imageURLs {
                if let result = loadImageAsMLXArray(imageURL) {
                    mlxArrays.append(result)
                    labels.append(label)
                }
            }
            return (mlxArrays, labels)
        } catch {
            print("Erro ao acessar os arquivos no diretório \(path): \(error)")
            return nil
        }
    }
    
    // Carregar imagens de quadrados (rótulo 1)
    guard let (quadradosImages, quadradosLabels) = loadImages(from: quadradosPath, label: 1) else {
        return nil
    }
    
    // Carregar imagens de não quadrados (rótulo 0)
    guard let (naoQuadradosImages, naoQuadradosLabels) = loadImages(from: naoQuadradosPath, label: 0) else {
        return nil
    }
    
    // Combinar as imagens e rótulos
    let images = quadradosImages + naoQuadradosImages
    let labels = quadradosLabels + naoQuadradosLabels
    
    // Converter labels inteiras para one-hot vectors
    _ = labels.map { label -> MLXArray in
        if label == 0 {
            return MLXArray(converting: [1.0, 0.0])  // Classe 0: [1, 0]
        } else {
            return MLXArray(converting: [0.0, 1.0])  // Classe 1: [0, 1]
        }
    }
    
    return (images, labels)
}


func loadImageAsMLXArray(_ imageURL: URL) -> (MLXArray)? {
    guard let imageSource = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
          let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
        print("Erro ao carregar imagem: \(imageURL.lastPathComponent)")
        return nil
    }
    
    let width = 28
    let height = 28
    let context = CGContext(
        data: nil,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: width,
        space: CGColorSpaceCreateDeviceGray(),
        bitmapInfo: CGImageAlphaInfo.none.rawValue
    )
    
    context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
    
    guard let pixelData = context?.data else {
        return nil
    }
    
    let buffer = pixelData.bindMemory(to: UInt8.self, capacity: width * height)
    let pixelArray = Array(UnsafeBufferPointer(start: buffer, count: width * height))
    
    var normalizedPixelArray = [Float](repeating: 0.0, count: width * height)
    for i in 0..<width * height {
        normalizedPixelArray[i] = Float(pixelArray[i]) / 255.0  // Normaliza para 0-1
    }
    
    let mlxArray = MLXArray(normalizedPixelArray, [1, 28, 28, 1])
    return (mlxArray)
}





class SimpleCNN: Module {
    var conv1: Conv2d
    var conv2: Conv2d
    var conv3: Conv2d
    var conv4: Conv2d
    var maxPool: MaxPool2d
    var fc1: Linear
    var fc2: Linear  // Nova camada MLP
    var fc3: Linear  // Camada de saída
    var dropout: Dropout
    override init() {
        // Camadas Convolucionais
        conv1 = Conv2d(inputChannels: 1, outputChannels: 32, kernelSize: 3, padding: 1)
        conv2 = Conv2d(inputChannels: 32, outputChannels: 64, kernelSize: 3, padding: 1)
        conv3 = Conv2d(inputChannels: 64, outputChannels: 128, kernelSize: 3, padding: 1)
        conv4 = Conv2d(inputChannels: 128, outputChannels: 256, kernelSize: 3, padding: 1)
        maxPool = MaxPool2d(kernelSize: 2, stride: 2)
        // MLP com duas camadas ocultas
        fc1 = Linear(256, 32)  // 256 features após flatten
        fc2 = Linear(32, 16)  // Nova camada intermediária
        fc3 = Linear(16, 2)    // Saída binária
        dropout = Dropout(p: 0.5)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        // Bloco 1
        x = conv1(x)
        x = relu(x)
        x = maxPool(x)
        // Bloco 2
        x = conv2(x)
        x = relu(x)
        x = maxPool(x)
        // Bloco 3
        x = conv3(x)
        x = relu(x)
        x = maxPool(x)
        // Bloco 4
        x = conv4(x)
        x = relu(x)
        x = maxPool(x)
        // Flatten e MLP
        x = x.reshaped([x.shape[0], 256])
        x = fc1(x)
        x = relu(x)
//        x = dropout(x)
        x = fc2(x)
        x = relu(x)
//        x = dropout(x)
        x = fc3(x)
//        x = relu(x)
        return x
    }
}

let optimizer = Adam(learningRate: 0.0001)


func binaryCrossEntropyLoss(predictions: MLXArray, targets: MLXArray) -> MLXArray {
    let epsilon: Float = 1e-7
    let clippedPredictions = clip(predictions, min: epsilon, max: 1.0 - epsilon)
    return -mean(targets * log(clippedPredictions) + (1 - targets) * log(1 - clippedPredictions))
}

func crossEntropyLoss(predictions: MLXArray, targets: MLXArray) -> MLXArray {
    let logSoftmax = log(softmax(predictions, axis: 1))
    return -mean(sum(targets * logSoftmax, axis: 1))
}

func lossFunction(model: SimpleCNN, input: MLXArray, target: MLXArray) -> MLXArray {
    let output = model(input)
    return crossEntropy(logits: output, targets: target, reduction: .mean)
}



func trainModel(model: SimpleCNN, images: [MLXArray], labels: [Int], epochs: Int, optimizer: Optimizer) {
    let lossAndGradients = valueAndGrad(model: model, lossFunction)
    let dataset = zip(images, labels).map { ($0, $1) }
    
    for epoch in 1...epochs {
        
        let shuffledDataset = dataset.shuffled()
        
        let batchsize = 10
            
        for startBatch in stride(from: 0, to: shuffledDataset.count, by: batchsize) {
            
            let end = min(startBatch + batchsize, shuffledDataset.count)
            let batch = Array(shuffledDataset[startBatch..<end])
            
            let batchImages = Array(batch.map {$0.0})
            let batchLabels = Array(batch.map {$0.1})
            
            let batchConcatenated = concatenated(batchImages)
            
            let oneHotLabel = batchLabels.map { label in
                label == 1 ? MLXArray(converting: [1.0, 0.0]) : MLXArray(converting: [0.0, 1.0])
            }
            
            let targetBatch = concatenated(oneHotLabel).reshaped([batchLabels.count, 2])
            
            let (loss, gradients) = lossAndGradients(model, batchConcatenated, targetBatch)
            
            optimizer.update(model: model, gradients: gradients)
            
            print("Epoch \(epoch)/\(epochs) - Loss: \(loss.item(Float.self))")
        }
    }
}


func testModel(model: SimpleCNN, testImages: [MLXArray], testLabels: [Int]) {
    var totalCorrect = 0
    let totalSamples = testImages.count

    for (image, label) in zip(testImages, testLabels) {
        let logits = model(image)
        let probabilities = softmax(logits, axis: 1) // Aplica softmax
//        print("Probabilities: \(probabilities)")
        // Pega a classe com maior probabilidade
        let predictedClass = probabilities.argMax().item(Int.self)
        
        print("\nImagem : Previsão = \(predictedClass), Alvo = \(label)")
        
        if predictedClass == label {
            totalCorrect += 1
        }
    }

    let accuracy = Float(totalCorrect) / Float(totalSamples)
    print("Acurácia do teste: \(accuracy * 100)% \n")
}



if let (images, labels) = loadDataset(quadradosPath: "dataset/quadrados", naoQuadradosPath: "dataset/outros") {
    // Criar o modelo
    let model = SimpleCNN()
    
//    let combine = zip(images,labels).shuffled()
//    let suffledImages = combine.map { $0.0 }
//    let shufflelabels = combine.map { $0.1 }
    trainModel(model: model, images: images, labels: labels, epochs: 200, optimizer: optimizer)
    
    
    if let (testImages, testLabels) = loadDataset(quadradosPath: "dataset/validacao", naoQuadradosPath: "dataset/validacaoTest") {
        let combined = zip(testImages, testLabels).shuffled()
        let shuffledImages = combined.map { $0.0 }
        let shuffledLabels = combined.map { $0.1 }
        testModel(model: model, testImages: shuffledImages, testLabels: shuffledLabels)
    }

    
    
} else {
    print("Falha ao carregar o dataset.")
}

