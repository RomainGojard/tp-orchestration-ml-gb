import unittest
from unittest.mock import patch, MagicMock
import src.kedro_road_sign as main_module

class TestMainEntryPoint(unittest.TestCase):
    @patch("kedro_road_sign.main.find_run_command")
    @patch("kedro_road_sign.main.configure_project")
    def test_main_calls_configure_and_run(self, mock_configure_project, mock_find_run_command):
        mock_run = MagicMock()
        mock_find_run_command.return_value = mock_run

        result = main_module.main("arg1", test_kwarg="value")

        # Check configure_project is called with correct package name
        expected_pkg = "kedro_road_sign"
        mock_configure_project.assert_called_once_with(expected_pkg)

        # Check find_run_command is called with the same package name
        mock_find_run_command.assert_called_once_with(expected_pkg)

        # Check that run was called with the args and kwargs (with standalone_mode added)
        mock_run.assert_called_once()
        called_args, called_kwargs = mock_run.call_args
        self.assertIn("standalone_mode", called_kwargs)
        self.assertEqual(called_kwargs["test_kwarg"], "value")

        # Check main returns the result of run()
        self.assertEqual(result, mock_run.return_value)

if __name__ == "__main__":
    unittest.main()
