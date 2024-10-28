import tkinter as tk
from gui import GibbsSamplerGUI

def main():
    root = tk.Tk()
    app = GibbsSamplerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
