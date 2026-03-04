"""
main.py — Application entry point.
"""

import sys
import pyqtgraph as pg

from main_window import MainWindow


def main():
    pg.mkQApp()
    win = MainWindow()
    win.resize(1300, 900)
    win.show()
    sys.exit(pg.exec())


if __name__ == "__main__":
    main()
