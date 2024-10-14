import os
import pandas as pd
from typing import List, Optional
import apikey

from rich.console import Console

console = Console()

def set_data_directory_path(path: str) -> None:
	"""
	Sets data directory path.

	:param path: Path to data directory
	"""
	apikey.save("ITTW_DATA_DIRECTORY_PATH", path)
	console.print(f'Informing the Third World data directory path set to {path}', style='bold blue')

def get_data_directory_path() -> str:
	"""
	Gets data directory path.

	:return: Data directory path
	"""
	return apikey.load("ITTW_DATA_DIRECTORY_PATH")

def read_csv_file(file_name: str, directory: Optional[str] = None, encodings: Optional[List[str]] = None, error_bad_lines: Optional[bool] = False) -> Optional[pd.DataFrame]:
	"""
	Reads a CSV file into a pandas DataFrame. This function allows specification of the directory, encodings, 
	and handling of bad lines in the CSV file. If the file cannot be read, the function returns None.

	:param file_name: String representing the name of the CSV file to read.
	:param directory: Optional string specifying the directory where the file is located. If None, it is assumed the file is in the current working directory.
	:param encodings: Optional list of strings specifying the encodings to try. Defaults to ['utf-8'].
	:param error_bad_lines: Optional boolean indicating whether to skip bad lines in the CSV. If False, an error is raised for bad lines. Defaults to False.
	:return: A pandas DataFrame containing the data from the CSV file, or None if the file cannot be read.
	"""
	# Set default encodings if none are provided
	if encodings is None:
		encodings = ['utf-8', 'latin1', 'iso-8859-1', 'utf-8-sig']

	# Read in the file
	file_path = file_name if directory is None else os.path.join(directory, file_name)
	
	# Try to read the file with each encoding
	for encoding in encodings:
		try:
			# Return the dataframe
			return pd.read_csv(file_path, low_memory=False, encoding=encoding, on_bad_lines='warn' if error_bad_lines else 'error')
		# If there's a Pandas error, print it and return None
		except pd.errors.EmptyDataError:
			console.print(f'Empty dataframe for {file_name}', style='bold red. Printed from function read_csv_file.')
			return None
		# If there's an encoding error, print it and try the next encoding
		except UnicodeDecodeError:
			console.print(f'Failed to read {file_name} with {encoding} encoding. Trying next encoding... Printed from function read_csv_file.', style='bold yellow')
		# If there's another type of error, print it and return None
		except Exception as e:
			console.print(f'Failed to read {file_name} with {encoding} encoding. Error: {e}. Printed from function read_csv_file.', style='bold red')
			return None

	# If none of the encodings worked, print an error and return None
	console.print(f'Failed to read {file_name} with any encoding. Printed from function read_csv_file.', style='bold red')
	return None

if __name__ == '__main__':
	set_data_directory_path('/Users/zleblanc/Informing-The-Third-World/periodical-collection-curation')
	data_directory_path = get_data_directory_path()
	console.print(f'Data directory path: {data_directory_path}', style='bold blue')