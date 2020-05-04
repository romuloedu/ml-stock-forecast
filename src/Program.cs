using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using StockForecast.Models;
using static Microsoft.ML.DataOperationsCatalog;
using static Microsoft.ML.Transforms.MissingValueReplacingEstimator;

namespace StockForecast
{
    class Program
    {
        private static string DatasetFile = @"../Data/PETR4.SA.csv";
        private static string BasePath = @"../MLModel";
        private static string ModelPath = $"{BasePath}/StockForecast.zip";

        static void Main(string[] args)
        {
            // Cria o contexto que trabalhará com aprendizado de máquina.
            MLContext context = new MLContext();

            // Lê o arquivo e o transforma em um dataset.
            TrainTestData splitData = Sanitize(context);

            ITransformer model = Train(context, splitData.TrainSet);

            RegressionMetrics metrics = Evaluate(context, model, splitData.TestSet);

            SaveModel(context, model, splitData.TrainSet.Schema);

            PrintMetrics(metrics);

            PredictPrice(context);

            Console.ReadLine();
        }

        private static TrainTestData Sanitize(MLContext context)
        {
            // Lê o arquivo e o transforma em um dataset.
            IDataView dataview = context.Data
            .LoadFromTextFile<StockInfo>(DatasetFile, ',', true);

            // Remove as linhas que contiverem algum valor nulo.
            dataview = context.Data.FilterRowsByMissingValues(dataview, "Open",
            "High", "Low", "AdjustedClose", "Volume");

            // Divide o dataset em uma base de treino (70%) e uma de teste (20%).
            TrainTestData trainTestData = context.Data.TrainTestSplit(dataview, 0.2);

            return trainTestData;
        }

        private static ITransformer Train(MLContext context, IDataView trainData)
        {
            SdcaRegressionTrainer sdcaTrainer = context.Regression.Trainers.Sdca();

            string[] featureColumns = { "Open", "High", "Low", "AdjustedClose", "Volume" };

            // Constroi o fluxo de transformação de dados e processamento do modelo.
            IEstimator<ITransformer> pipeline = context.Transforms
            .CopyColumns("Label", "Close")
            .Append(context.Transforms.Concatenate("Features", featureColumns))
            .Append(context.Transforms.NormalizeMinMax("Features"))
            .AppendCacheCheckpoint(context)
            .Append(sdcaTrainer);

            ITransformer model = pipeline.Fit(trainData);

            return model;
        }

        private static RegressionMetrics Evaluate(MLContext context, ITransformer model,
        IDataView testSet)
        {
            IDataView predictions = model.Transform(testSet);

            RegressionMetrics metrics = context.Regression.Evaluate(predictions);

            return metrics;
        }

        private static void SaveModel(MLContext context, ITransformer model,
        DataViewSchema schema)
        {
            if (!Directory.Exists(BasePath))
            {
                Directory.CreateDirectory(BasePath);
            }
            else
            {
                foreach (String file in Directory.EnumerateFiles(BasePath))
                {
                    File.Delete(file);
                }
            }

            context.Model.Save(model, schema, ModelPath);
        }

        private static void PrintMetrics(RegressionMetrics metrics)
        {
            Console.WriteLine("-------------------- MÉTRICAS --------------------");
            Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError}");
            Console.WriteLine($"Mean Squared Error: {metrics.MeanSquaredError}");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");
            Console.WriteLine($"R Squared: {metrics.RSquared}");
            Console.WriteLine("--------------------------------------------------");
        }

        private static void PredictPrice(MLContext context)
        {
            throw new NotImplementedException();
        }
    }
}
