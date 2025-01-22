import unittest
from curve2flood.core import read_input_file, Curve2Flood_MainFunction

class TestCurve2Flood(unittest.TestCase):
    def test_read_input_file(self):
        # Create a mock input file
        input_content = """
        DEM_File path/to/dem.tif
        Stream_File path/to/strm.tif
        LU_Raster_SameRes path/to/land.tif
        StrmShp_File path/to/streams.shp
        OutFLD path/to/output_flood.tif
        LU_Water_Value  80
        Q_Fraction 0.5
        TopWidthPlausibleLimit 200
        TW_MultFact 1.0
        Set_Depth 0.1
        LocalFloodOption True
        Flood_WaterLC_and_STRM_Cells False
        """
        with open("test_input.txt", "w") as f:
            f.write(input_content)
        
        # Test the input file reading
        params = read_input_file("test_input.txt")
        self.assertEqual(params["DEM_File"], "path/to/dem.tif")
        self.assertEqual(params["Q_Fraction"], "0.5")
        self.assertTrue(params["LocalFloodOption"])
    
    def test_curve2flood_main_function(self):
        # Create a mock input file
        input_content = """
        DEM_File path/to/dem.tif
        Stream_File path/to/strm.tif
        LU_Raster_SameRes path/to/land.tif
        StrmShp_File path/to/streams.shp
        OutFLD path/to/output_flood.tif
        LU_Water_Value  80
        Q_Fraction 0.5
        TopWidthPlausibleLimit 200
        TW_MultFact 1.0
        Set_Depth 0.1
        LocalFloodOption True
        Flood_WaterLC_and_STRM_Cells False
        """
        with open("test_input_main.txt", "w") as f:
            f.write(input_content)
        
        # Call the main function and verify no exceptions are raised
        try:
            Curve2Flood_MainFunction("test_input_main.txt")
        except Exception as e:
            self.fail(f"Curve2Flood_MainFunction raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()