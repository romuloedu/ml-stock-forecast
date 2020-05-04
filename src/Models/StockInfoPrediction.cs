using System;
using Microsoft.ML.Data;

namespace StockForecast.Models
{
    public class StockInfoPrediction
    {
        [ColumnName("Score")]
        public float Close { get; set; }
    }
}