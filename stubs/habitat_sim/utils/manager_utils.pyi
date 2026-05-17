from __future__ import annotations
import csv as csv
__all__: list[str] = ['csv', 'save_csv_report']
def save_csv_report(file_name: str, report_string: str):
    """
    This function takes a string, parses it on embedded newlines,
        and writes the resultant array of CSV strings to a file.  This is the output format
        of the various Attributes Managers and Physics Object manager's reports.
    
        :param file_name: The name of the file to write to.
        :param report_string: String with embedded newlines to par to write to file.
    """
