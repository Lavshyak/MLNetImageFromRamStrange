using ImageFromRamStrange;
using Microsoft.ML;
using Microsoft.ML.Data;
using SkiaSharp;

var config = new Config(2, 2, 1);

var mlContext = new MLContext();

var pipeline =
    mlContext.Transforms.ResizeImages(
            inputColumnName: "SourceImage", outputColumnName: "ResizedImage",
            imageHeight: config.TargetImageHeight, imageWidth: config.TargetImageWidth)
        .Append(
            mlContext.Transforms.ExtractPixels(
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
    new { Bitmap = CreateSkBitmap(), LabelValue = "1" },
    new { Bitmap = CreateSkBitmap(), LabelValue = "2" },
    new { Bitmap = CreateSkBitmap(), LabelValue = "3" },
};

var trainingData = bitmapsAndPathsAndLabels.Select(b =>
{
    var skImage = SKImage.FromBitmap(b.Bitmap);
    var imageBytes = skImage.Encode(SKEncodedImageFormat.Jpeg, 100).ToArray();
    using var stream = new MemoryStream(imageBytes);
    var mlImage = MLImage.CreateFromStream(stream);
    
    return new DataModels.DataInput()
        { SourceImage = mlImage, LabelValue = b.LabelValue };
    
}).ToArray();

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
    var bitmap = new SKBitmap(8, 8);
    var newPixels = bitmap.Pixels.Select(p => new SKColor((uint)Random.Shared.Next())).ToArray();
    bitmap.Pixels = newPixels;
    return bitmap;
}

public record Config(
    int TargetImageHeight,
    int TargetImageWidth,
    int RandomSeed
);