namespace ImageFromFileOK;

public class DataModels
{
    public class DataInput
    {
        public string ImagePath { get; set; } = null!;
        public string LabelValue { get; set; } = null!;
    }
    public class DataOutput : DataInput
    {
        public string PredictedLabelValue { get; set; } = null!;
    }
}