

import sys
import pyqtgraph as pg

from main_window import MainWindow


def main():
    pg.mkQApp()
    win = MainWindow()
    win.resize(1400, 1000)
    win.show()
    sys.exit(pg.exec())


if __name__ == "__main__":
    main()