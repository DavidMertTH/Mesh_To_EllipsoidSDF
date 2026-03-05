

import sys

import pyqtgraph as pg

from main_window import MainWindow


def main():
    pg.mkQApp()
    win = MainWindow()
    win.resize(1400, 1000)
    win.show()
    win.start_optimization(method="adam", num_steps=2000, report_every=20)
    sys.exit(pg.exec())


if __name__ == "__main__":
    main()