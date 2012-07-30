# -*- coding: utf-8 -*-
"""
Created on Jul 29, 2012

@author: marko
"""
from PIL import Image, ImageTk
from Tkinter import Frame, Label, Button, Toplevel
from pvmodule_tk import PVmodule_tk
from pvstring_tk import PVstring_tk
from pvsystem_tk import PVsystem_tk
import os


SPLOGO = os.path.join('res', 'logo_bg.png')
PVMODULE_TEXT = 'PVmodule'
PVSTRING_TEXT = 'PVstring'
PVSYSTEM_TEXT = 'PVsystem'


class PVapplicaton(Frame):
    """
    classdocs
    """

    def __init__(self, master=None):
        """
        Constructor
        """
        Frame.__init__(self, master)
        self.pack(expand=True)  # if user resizes, expand Frame
        self.pack(fill='both')

        self['bg'] = 'black'  # set black background
        self['padx'] = '15'  # pad sides with 15 points
        self['pady'] = '5'  # pad top/bottom 5 points
        self.master.title('PVmismatch')  # set title bar
        self.image = Image.open(SPLOGO)  # create image object
        # convert image to tk-compatible format
        self.SPlogo = ImageTk.PhotoImage(self.image)
        self.SPlogoLabel = Label(self, image=self.SPlogo,
                                 cnf={'borderwidth': '0'})
        self.SPlogoLabel.pack({'side': 'top'})

        self.PVmoduleButton = Button(self, cnf={'text': PVMODULE_TEXT})
        self.PVmoduleButton.pack({'side': 'top', 'fill': 'both'})
        self.PVmoduleButton['command'] = self.startPVmodule_tk

        self.PVstringButton = Button(self, cnf={'text': PVSTRING_TEXT})
        self.PVstringButton.pack({'side': 'top', 'fill': 'both'})
        self.PVstringButton['command'] = self.startPVstring_tk

        self.PVsystemButton = Button(self, cnf={'text': PVSYSTEM_TEXT})
        self.PVsystemButton.pack({'side': 'top', 'fill': 'both'})
        self.PVsystemButton['command'] = self.startPVsystem_tk

        self.QUIT = Button(self, cnf={'text': 'Quit', 'command': self.quit})
        self.QUIT.pack({'side': 'top', 'fill': 'both'})

    def startPVmodule_tk(self):
        top = Toplevel()
        app = PVmodule_tk(self, top)
        app.mainloop()
        # please destroy me or I'll continue to run in background
        top.destroy()

    def startPVstring_tk(self):
        top = Toplevel()
        app = PVstring_tk(self, top)
        app.mainloop()
        # please destroy me or I'll continue to run in background
        top.destroy()

    def startPVsystem_tk(self):
        top = Toplevel()
        app = PVsystem_tk(self, top)
        app.mainloop()
        # please destroy me or I'll continue to run in background
        top.destroy()
