using KelpNet;
using KelpNet.CL;
using System.Collections.Generic;

namespace ConvolutionChecker
{
    class Program
    {
        static void Main(string[] args)
        {
            RunAsync();
        }

        private static void RunAsync()
        {
            var trainData = new NdArray(new[] { 2, 3, 4 });
            for (int i = 0; i < trainData.Data.Length; i++)
            {
                trainData.Data[i] = (float)i / trainData.Data.Length;
            }

            var functions = new List<Function>();
            functions.Add(new Convolution2D(2, 1, 3));

            var nn = new FunctionStack(functions.ToArray());
            nn.Compress();
            var optimizer = new Adam();
            nn.SetOptimizer(optimizer);

            var result = nn.Predict(trainData)[0];
        }
    }
}
