import unittest
from curve2flood.core import get_params, Curve2Flood_MainFunction

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
        TopWidthPlausibleLimit 200
        TW_MultFact 1.0
        Set_Depth 0.1
        LocalFloodOption True
        Flood_WaterLC_and_STRM_Cells False
        """
        with open("test_input.txt", "w") as f:
            f.write(input_content)
        
        # Test the input file reading
        params = get_params("test_input.txt")
        self.assertEqual(params["DEM_File"], "path/to/dem.tif")
        self.assertEqual(params["TW_MultFact"], 1.0)
        self.assertTrue(params["LocalFloodOption"])
        self.assertEqual(params["Set_Depth"], 0.1)

    def test_set_depth_keys(self):
        # Either key sets the depth (FloodSpreader_SpecifyDepth is what nencarta writes); without one, depths come from flows
        self.assertEqual(get_params(args={"FloodSpreader_SpecifyDepth": 0.3})["Set_Depth"], 0.3)
        self.assertEqual(get_params(args={"Set_Depth": "0.2", "FloodSpreader_SpecifyDepth": 0.3})["Set_Depth"], 0.2)
        self.assertLess(get_params(args={"DEM_File": "path/to/dem.tif"})["Set_Depth"], 0.0)

    def test_curve2flood_main_function(self):
        # Create a mock input file
        input_content = """
        DEM_File path/to/dem.tif
        Stream_File path/to/strm.tif
        LU_Raster_SameRes path/to/land.tif
        StrmShp_File path/to/streams.shp
        OutFLD path/to/output_flood.tif
        LU_Water_Value  80
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
            if not isinstance(e, RuntimeError) or "No such file or directory" not in str(e):
                self.fail(f"Curve2Flood_MainFunction raised an exception: {e.with_traceback(e.__traceback__)}")

if __name__ == "__main__":
    unittest.main()