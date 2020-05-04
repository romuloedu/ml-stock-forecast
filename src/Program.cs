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

            PredictPrice(context, model);

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
            var trainer = context.Regression.Trainers.Sdca();

            string[] featureColumns = { "Open", "High", "Low", "Volume" };

            // Constroi o fluxo de transformação de dados e processamento do modelo.
            IEstimator<ITransformer> pipeline = context.Transforms
            .CopyColumns("Label", "Close")
            .Append(context.Transforms.Concatenate("Features", featureColumns))
            .Append(context.Transforms.NormalizeMinMax("Features"))
            .AppendCacheCheckpoint(context)
            .Append(trainer);

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

        private static void PredictPrice(MLContext context, ITransformer model)
        {
            StockInfo[] stocks = {
                new StockInfo
                {
                    Open = 25.700001f,
                    High = 25.780001f,
                    Low = 25.430000f,
                    Close = 25.450001f,
                    AdjustedClose = 21.730824f,
                    Volume = 17841800
                },
                new StockInfo
                {
                    Open = 30.799999f,
                    High = 30.889999f,
                    Low = 29.750000f,
                    Close = 29.920000f,
                    AdjustedClose = 29.918381f,
                    Volume = 73522900
                },
                new StockInfo
                {
                    Open = 16.670000f,
                    High = 16.760000f,
                    Low = 15.530000f,
                    Close = 15.720000f,
                    AdjustedClose = 15.719150f,
                    Volume = 115633300
                },
                new StockInfo
                {
                    Open = 17.51f,
                    High = 17.62f,
                    Low = 17.20f,
                    Close = 17.25f,
                    AdjustedClose = 0f,
                    Volume = 0
                }
            };

            PredictionEngine<StockInfo, StockInfoPrediction> predictor = context.Model
            .CreatePredictionEngine<StockInfo, StockInfoPrediction>(model);

            foreach (StockInfo stock in stocks)
            {
                StockInfoPrediction prediction = predictor.Predict(stock);

                Console.WriteLine("---------------- PREVISÃO ----------------");
                Console.WriteLine($"O preço previsto para a ação é de R$ {prediction.Close:0.#0}");
                Console.WriteLine($"O preço atual é de R$ {stock.Close:0.#0}");
                Console.WriteLine($"Diferença de R$ {prediction.Close - stock.Close:0.#0}");
                Console.WriteLine("------------------------------------------");
            }
        }
    }
}
