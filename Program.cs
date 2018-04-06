using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;

using OSGeo.GDAL;
using OSGeo.OGR;
using System.IO;


namespace LakeExtraction
{
    class Program
    { 
        static void Main(string[] args)
        {

            BandRatioTest();
            ExtractLargeStreams_BatchProcess();    
        }


        /// <summary>
        /// calculate ndwi image
        /// </summary>
        static void BandRatioTest()
        {
            // string intputDirectoryName = @"D:\Analysis\Greenland_Analysis\GreenlandHighRes\StreamExtraction\multiinput\";
            string intputDirectoryName = @"F:\Courtney\Greenland_Code\Courtney_Stream_Extraction\multiinput\";
            //@"D:\2012Images\WorldView\geotiff\718\";
            // string outputDirectoryName = @"D:\Analysis\Greenland_Analysis\GreenlandHighRes\StreamExtraction\multiinput\ndwi\";
            string outputDirectoryName = @"F:\Courtney\Greenland_Code\Courtney_Stream_Extraction\multioutput\ndwi\";
            //@"D:\2012Images\WorldView\geotiff\718\ndwi\";

            string[] files = System.IO.Directory.GetFiles(intputDirectoryName);
            for (int i = 0; i < files.Length; i++)
            {
                if (System.IO.Path.GetExtension(files[i]) == ".tif")
                {
                    LakeExtraction worldView2LakeExtraction = new LakeExtraction(files[i], SensorType.WorldView2);
                    Console.WriteLine("start process: {0}", files[i]);

                    Stopwatch sw = new Stopwatch();
                    sw.Start();

                    float[] ratioData = worldView2LakeExtraction.BandRatioCalculate(2, 8);
                    worldView2LakeExtraction.ExportData(ratioData, outputDirectoryName + System.IO.Path.GetFileNameWithoutExtension(files[i]) + "_normalized_ratio_blue2_nir8_ratio.img");

                    sw.Stop();
                    int time = Convert.ToInt32(sw.ElapsedMilliseconds / 1000);
                    Console.WriteLine("Done...cost {0} second", time);
                    Console.WriteLine();
                }
            }
        }

        static void ExtractLargeStreams_BatchProcess()
        {
            // string intputDirectoryName = @"D:\Analysis\Greenland_Analysis\GreenlandHighRes\StreamExtraction\multiinput\ndwi\";
            // string intputDirectoryName = @"F:\Courtney\Greenland_Code\Courtney_Stream_Extraction\multiinput\ndwi\";
            string intputDirectoryName = @"F:\Courtney\Greenland_Code\Courtney_Stream_Extraction\multioutput\ndwi\";
            string[] files = System.IO.Directory.GetFiles(intputDirectoryName);
            for (int i = 0; i < files.Length; i++)
            {
                if (System.IO.Path.GetExtension(files[i]) == ".img" && files[i].Contains("normalized_ratio_blue2_nir8"))
                {
                    Console.WriteLine("-------------------------------------------------");
                    Console.WriteLine("start process: {0}", files[i]);
                    ExtractLargeStreams(files[i]);
                    Console.WriteLine("--------------------END--------------------------");
                }

                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }

        //only extract large streams
        static void ExtractLargeStreams(string inputFileName)
        {
            StreamExtraction se = new StreamExtraction(inputFileName);

            string outputFileName = (System.IO.Path.GetFileNameWithoutExtension(inputFileName).Split('-'))[0];

            // string path = System.IO.Path.GetDirectoryName(inputFileName) + "\\";
            string path = @"F:\Courtney\Greenland_Code\Courtney_Stream_Extraction\multioutput\watermask\";

            //increase this threshold for extracting lakes ~1.55
            //for streams, usually ~1.25; change for each strip
            float globalThreshold = 1.25f;
            string gt = Convert.ToString(globalThreshold).Split('.')[0] + Convert.ToString(globalThreshold).Split('.')[1];

            //increase this threshold for extracting lakes, ~5000, something very large
            //for streams, usually ~500, change for each strip
            int sizeThrshold = 500;
            string outputLargeRiverFileName_withoutFilter = path + outputFileName + "_" + Convert.ToString(gt) + "_" + Convert.ToString(sizeThrshold) + "_watermask.img";

            se.ExtractLargeStreams(globalThreshold, sizeThrshold, outputLargeRiverFileName_withoutFilter);

            GC.Collect();
            GC.WaitForPendingFinalizers();
        }


        /// <summary>
        /// process ndwi image and extract streams
        /// </summary>
        static void StreamExtractionTest4LargeImage_BatchProcess()
        {
            string intputDirectoryName = @"D:\Analysis\Greenland_Analysis\GreenlandHighRes\StreamExtraction\output_ndwi\";

            string[] files = System.IO.Directory.GetFiles(intputDirectoryName);
            for (int i = 0; i < files.Length; i++)
            {
                if (System.IO.Path.GetExtension(files[i]) == ".img" && files[i].Contains("normalized_ratio_blue2_nir8"))
                {
                    Console.WriteLine("-------------------------------------------------");
                    Console.WriteLine("start process: {0}", files[i]);
                    
                    //process ndwi image
                    StreamExtraction se = new StreamExtraction(files[i]);
                    se.ProcessLargeImage1Spectral2Edge(1.25f, EdgeDetector.Canny, 20, 40, 5, 7, 500, 1.10f, 5, 10, 1.45f);


                    Console.WriteLine("--------------------END--------------------------");
                }
            }
        }

    }
}
