using ImageFromFileOK;
using Microsoft.ML;
using SkiaSharp;

var config = new Config(2, 2, 1, Path.Combine(Environment.CurrentDirectory, "images"));

var mlContext = new MLContext();

var pipeline =
    mlContext.Transforms.LoadImages(
            inputColumnName: nameof(DataModels.DataInput.ImagePath), outputColumnName: "SourceImage",
            imageFolder: "")
        .Append(mlContext.Transforms.ResizeImages(
            inputColumnName: "SourceImage", outputColumnName: "ResizedImage",
            imageHeight: config.TargetImageHeight, imageWidth: config.TargetImageWidth))
        .Append(mlContext.Transforms.ExtractPixels(
            inputColumnName: "ResizedImage", outputColumnName: "ExtractedPixels",
            interleavePixelColors: true, offsetImage: 177))
        .Append(mlContext.Transforms.Conversion.MapValueToKey(
            inputColumnName: nameof(DataModels.DataInput.LabelValue), outputColumnName: "LabelKey"))
        .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(
            labelColumnName: "LabelKey", featureColumnName: "ExtractedPixels"
            /*outputColumnName: "PredictedLabel"*/))
        .Append(mlContext.Transforms.Conversion.MapKeyToValue(
            inputColumnName: "PredictedLabel", outputColumnName: nameof(DataModels.DataOutput.PredictedLabelValue)))
        .AppendCacheCheckpoint(mlContext);

var bitmapsAndPathsAndLabels = new[]
{
    new {Bitmap = CreateSkBitmap(), FilePath = Path.Combine(config.ImagesRoot, "1.jpg"), LabelValue = "1"},
    new {Bitmap = CreateSkBitmap(), FilePath = Path.Combine(config.ImagesRoot, "2.jpg"), LabelValue = "2"},
    new {Bitmap = CreateSkBitmap(), FilePath = Path.Combine(config.ImagesRoot, "3.jpg"), LabelValue = "3"},
};
Directory.CreateDirectory(config.ImagesRoot);
foreach (var bitmapAndPathAndLabel in bitmapsAndPathsAndLabels)
{
    File.WriteAllBytes(bitmapAndPathAndLabel.FilePath, SKImage.FromBitmap(bitmapAndPathAndLabel.Bitmap).Encode(SKEncodedImageFormat.Jpeg, 100).ToArray());
}
var trainingData = bitmapsAndPathsAndLabels.Select(b => new DataModels.DataInput()
    { ImagePath = b.FilePath, LabelValue = b.LabelValue }).ToArray();
var trainingDataView = mlContext.Data.LoadFromEnumerable(trainingData);

var model = pipeline.Fit(trainingDataView);

var transformedData = model.Transform(trainingDataView);

var predictionEngine = mlContext.Model.CreatePredictionEngine
    <DataModels.DataInput, DataModels.DataOutput>
    (model);

var predictionResult = predictionEngine.Predict(trainingData[0]);
Console.WriteLine(predictionResult.PredictedLabelValue);
Console.WriteLine("end.");

SKBitmap CreateSkBitmap()
{
    var bitmap = new SKBitmap(8,8);
    var newPixels = bitmap.Pixels.Select(p => new SKColor((uint)Random.Shared.Next())).ToArray();
    bitmap.Pixels = newPixels;
    return bitmap;
}

public record Config(
    int TargetImageHeight,
    int TargetImageWidth,
    int RandomSeed,
    string ImagesRoot
);