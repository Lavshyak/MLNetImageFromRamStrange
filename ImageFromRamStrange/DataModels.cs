using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

namespace ImageFromRamStrange;

public class DataModels
{
    public class DataInput
    {
        public string LabelValue { get; set; } = null!;
        
        [ImageType(1,1)] // the actual size is not important when it is passed to mlContext.Transforms.ResizeImages
        public MLImage SourceImage { get; set; } = null!;
    }

    public class DataOutput : DataInput
    {
        public string PredictedLabelValue { get; set; } = null!;
    }
}