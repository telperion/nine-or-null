import csv
from datetime import datetime as dt, timedelta as td
import logging
import os
import re

import matplotlib
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np

import wx
import wx.adv
import wx.grid

from . import batch_process, guess_paradigm, plot_fingerprint, _VERSION, _CSV_FIELDNAMES

class AboutWithLinks(wx.Dialog):
    def __init__(self, *args, **kwargs):
        super(AboutWithLinks, self).__init__(*args, **kwargs)

        self.SetSize((540, 300))

        label_preceding = wx.StaticText(self,
            label=re.sub('\n[ \t]+', '\n', """+9ms or Null? is a StepMania simfile unbiasing utility.

            This utility can determine whether the sync bias of a simfile or a pack is +9ms (In The Groove) or null (general StepMania). A future version will also offer to unify it under one of those two options.
            It is not meant to perform a millisecond-perfect sync!

            You can read more about the origins of the +9ms sync bias here:"""),
        )
        href_cfwiki = wx.adv.HyperlinkCtrl(self,
            label="Club Fantastic Wiki's explanation",
            url="https://wiki.clubfantastic.dance/Sync#itg-offset-and-the-9ms-bias"
        )
        href_meowgarden = wx.adv.HyperlinkCtrl(self,
            label="Ash's discussion of solutions @ meow.garden",
            url="https://meow.garden/killing-the-9ms-bias"
        )
        label_following = wx.StaticText(self,
            label=re.sub('\n[ \t]+', '\n', f"""
            +9ms or Null? v{_VERSION}
            Sync bias algorithm and program written by Telperion.
            Credit to beware for sprouting the idea of an aligned audio fingerprint to examine sync.""")
        )
        button_ok = wx.Button(self, id=wx.ID_OK)

        label_preceding.Wrap(480)
        label_following.Wrap(480)
        self.Bind(wx.EVT_BUTTON, self.OnExit, button_ok)

        sizer = wx.GridBagSizer(3, 6)
        sizer.Add(wx.StaticBitmap(self, id=wx.ID_INFO, bitmap=wx.ArtProvider.GetBitmap(wx.ART_INFORMATION)),
                                   pos=(0, 0), span=(5, 1))
        sizer.Add(label_preceding, pos=(0, 1))
        sizer.Add(href_cfwiki,     pos=(1, 1))
        sizer.Add(href_meowgarden, pos=(2, 1))
        sizer.Add(label_following, pos=(3, 1))
        sizer.Add(button_ok,       pos=(4, 1))
        
        sizer_v = wx.BoxSizer(wx.VERTICAL)
        sizer_h = wx.BoxSizer(wx.HORIZONTAL)
        sizer_h.Add(sizer,   1, wx.CENTER)
        sizer_v.Add(sizer_h, 1, wx.CENTER)
        self.SetSizer(sizer_v)
        self.Layout()
    
    def OnExit(self, event):
        self.Destroy()


class NineOrNull(wx.Frame):
    def __init__(self, *args, **kwargs):
        super(NineOrNull, self).__init__(*args, **kwargs)

        self.SetSize(540, 720)
        self.panel_main = wx.Panel(self)

        # --------------------------------------------------------------
        # Paths
        self.label_root = wx.StaticText(self.panel_main, label='Path to simfile(s):')
        self.entry_root = wx.TextCtrl(self.panel_main, value=r'C:\Games\ITGmania\Songs')
        self.entry_root.SetToolTip(wx.ToolTip('Choose the path to the simfile, pack, or group of packs you wish to analyze.'))
        self.entry_root.SetMinSize((360, 24))
        self.button_root = wx.BitmapButton(self.panel_main, bitmap=wx.ArtProvider.GetBitmap(wx.ART_FOLDER_OPEN))
        self.button_root.SetToolTip(wx.ToolTip('Navigate to a simfile/pack directory...'))
        self.label_report_path = wx.StaticText(self.panel_main, label='Path to bias report:')
        self.button_report_path_auto = wx.BitmapToggleButton(self.panel_main)
        self.button_report_path_auto.SetToolTip(wx.ToolTip('Automatically choose a directory under the path above for reports and plots.'))
        self.button_report_path_auto.SetBitmap(wx.ArtProvider.GetBitmap(wx.ART_GO_UP))
        self.button_report_path_auto.SetValue(True)
        self.entry_report_path = wx.TextCtrl(self.panel_main, value=r'C:\Games\ITGmania\Songs\__bias-check')
        self.entry_report_path.SetToolTip(wx.ToolTip('Choose a destination for the sync bias report and audio fingerprint plots.'))
        self.entry_report_path.Disable()
        self.button_report_path = wx.BitmapButton(self.panel_main, bitmap=wx.ArtProvider.GetBitmap(wx.ART_FOLDER_OPEN))
        self.button_report_path.SetToolTip(wx.ToolTip('Navigate to a report/plot directory...'))
        self.button_report_path.Disable()

        for button in [self.button_root, self.button_report_path_auto, self.button_report_path]:
            button.SetMinSize((28, 28))

        sizer_paths = wx.GridBagSizer(6, 12)
        sizer_paths.Add(self.label_root,              (0, 0),              flag=wx.EXPAND | wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
        sizer_paths.Add(self.entry_root,              (0, 1), span=(1, 2), flag=wx.EXPAND | wx.ALIGN_CENTER_VERTICAL)
        sizer_paths.Add(self.button_root,             (0, 3),              flag=wx.EXPAND | wx.CENTER)
        sizer_paths.Add(self.label_report_path,       (1, 0),              flag=wx.EXPAND | wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
        sizer_paths.Add(self.button_report_path_auto, (1, 1),              flag=wx.EXPAND | wx.CENTER)
        sizer_paths.Add(self.entry_report_path,       (1, 2),              flag=wx.EXPAND | wx.ALIGN_CENTER_VERTICAL)
        sizer_paths.Add(self.button_report_path,      (1, 3),              flag=wx.EXPAND | wx.CENTER)
        sizer_paths.AddGrowableCol(2, 1)
        
        # --------------------------------------------------------------
        # Parameters
        self.panel_paradigms = wx.Panel(self.panel_main)
        self.panel_paradigms.SetBackgroundColour(self.GetBackgroundColour().ChangeLightness(170))
        self.panel_paradigms.SetMinSize((150, 120))
        self.panel_fingerprint = wx.Panel(self.panel_main)
        self.panel_fingerprint.SetBackgroundColour(self.GetBackgroundColour().ChangeLightness(170))
        self.panel_fingerprint.SetMinSize((186, 120))
        self.panel_attack = wx.Panel(self.panel_main)
        self.panel_attack.SetBackgroundColour(self.GetBackgroundColour().ChangeLightness(170))
        self.panel_attack.SetMinSize((168, 120))
        
        # - Sync bias paradigm parameters
        self.label_paradigms = wx.StaticText(     self.panel_paradigms, label='Sync bias paradigms')
        self.label_paradigms.SetFont(self.label_paradigms.GetFont().MakeUnderlined())
        self.checkbox_null   = wx.CheckBox(       self.panel_paradigms, label='Null (StepMania)')
        self.checkbox_null.SetToolTip(wx.ToolTip('Consider charts close enough to 0ms bias to be "correct" under the null sync paradigm.'))
        self.checkbox_null.SetValue(True)
        self.checkbox_p9ms   = wx.CheckBox(       self.panel_paradigms, label='+9ms (In The Groove)')
        self.checkbox_p9ms.SetToolTip(wx.ToolTip('Consider charts close enough to +9ms bias to be "correct" under the ITG sync paradigm.'))
        self.checkbox_p9ms.SetValue(True)
        self.label_tolerance = wx.StaticText(     self.panel_paradigms, label='Tolerance (ms): ±')
        self.spin_tolerance   = wx.SpinCtrlDouble(self.panel_paradigms, style=wx.SP_ARROW_KEYS, min=0.5, max=4.5, initial=3, inc=0.5)
        self.spin_tolerance.SetDigits(1)
        
        sizer_paradigms = wx.GridBagSizer(6, 0)
        sizer_paradigms.Add(self.label_paradigms, (0, 0), span=(1, 2), flag=wx.EXPAND | wx.ALIGN_CENTER)
        sizer_paradigms.Add(self.checkbox_null,   (1, 0), span=(1, 2), flag=wx.EXPAND | wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_LEFT)
        sizer_paradigms.Add(self.checkbox_p9ms,   (2, 0), span=(1, 2), flag=wx.EXPAND | wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_LEFT)
        sizer_paradigms.Add(self.label_tolerance, (3, 0),              flag=wx.EXPAND | wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
        sizer_paradigms.Add(self.spin_tolerance,  (3, 1),              flag=wx.EXPAND | wx.CENTER)
        sizer_paradigms.AddGrowableCol(0, 1)
        sizer_v = wx.BoxSizer(wx.VERTICAL)
        sizer_h = wx.BoxSizer(wx.HORIZONTAL)
        sizer_h.Add(sizer_paradigms, 1, wx.CENTER)
        sizer_v.Add(sizer_h,         1, wx.CENTER)
        self.panel_paradigms.SetSizer(sizer_v)

        # - Audio fingerprint parameters
        self.label_fingerprint      = wx.StaticText(    self.panel_fingerprint, label='Audio fingerprint')
        self.label_fingerprint.SetFont(self.label_fingerprint.GetFont().MakeUnderlined())
        self.label_fingerprint_size = wx.StaticText(    self.panel_fingerprint, label='Fingerprint size (ms): ±')
        self.spin_fingerprint_size  = wx.SpinCtrl(      self.panel_fingerprint, style=wx.SP_ARROW_KEYS, min=30, max=100, initial=50)
        self.spin_fingerprint_size.SetToolTip(wx.ToolTip('Time margin on either side of the beat to analyze.'))
        self.label_window_size      = wx.StaticText(    self.panel_fingerprint, label='Spectral window (ms): ±')
        self.spin_window_size       = wx.SpinCtrl(      self.panel_fingerprint, style=wx.SP_ARROW_KEYS, min=5, max=20, initial=10)
        self.spin_window_size.SetToolTip(wx.ToolTip('The spectrogram algorithm\'s moving window parameter.'))
        self.label_step_size        = wx.StaticText(    self.panel_fingerprint, label='Spectral step (ms): ±')
        self.spin_step_size         = wx.SpinCtrlDouble(self.panel_fingerprint, style=wx.SP_ARROW_KEYS, min=0.05, max=1.0, initial=0.2, inc=0.05)
        self.spin_step_size.SetToolTip(wx.ToolTip('Controls the spectrogram algorithm\'s overlap parameter, but expressed as a step size.'))
        self.spin_step_size.SetDigits(2)
        
        sizer_fingerprint = wx.GridBagSizer(6, 0)
        sizer_fingerprint.Add(self.label_fingerprint,      (0, 0), span=(1, 2), flag=wx.EXPAND | wx.ALIGN_CENTER)
        sizer_fingerprint.Add(self.label_fingerprint_size, (1, 0),              flag=wx.EXPAND | wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
        sizer_fingerprint.Add(self.spin_fingerprint_size,  (1, 1),              flag=wx.EXPAND | wx.CENTER)
        sizer_fingerprint.Add(self.label_window_size,      (2, 0),              flag=wx.EXPAND | wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
        sizer_fingerprint.Add(self.spin_window_size,       (2, 1),              flag=wx.EXPAND | wx.CENTER)
        sizer_fingerprint.Add(self.label_step_size,        (3, 0),              flag=wx.EXPAND | wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
        sizer_fingerprint.Add(self.spin_step_size,         (3, 1),              flag=wx.EXPAND | wx.CENTER)
        sizer_fingerprint.AddGrowableCol(0, 1)
        sizer_v = wx.BoxSizer(wx.VERTICAL)
        sizer_h = wx.BoxSizer(wx.HORIZONTAL)
        sizer_h.Add(sizer_fingerprint, 1, wx.CENTER)
        sizer_v.Add(sizer_h,           1, wx.CENTER)
        self.panel_fingerprint.SetSizer(sizer_v)

        # - Attack analysis parameters
        self.label_attack           = wx.StaticText(    self.panel_attack, label='Attack analysis')
        self.label_attack.SetFont(self.label_attack.GetFont().MakeUnderlined())
        self.label_kernel_target    = wx.StaticText(    self.panel_attack, label='Examine: ')
        self.combo_kernel_target    = wx.ComboBox(      self.panel_attack, style=wx.CB_DROPDOWN | wx.CB_READONLY, value='Beat digest', choices=['Beat digest', 'Accumulator'])
        self.combo_kernel_target.SetToolTip(wx.ToolTip('Choose whether to convolve with the beat digest or the spectral accumulator.'))
        self.label_kernel_type      = wx.StaticText(    self.panel_attack, label='Kernel: ')
        self.combo_kernel_type      = wx.ComboBox(      self.panel_attack, style=wx.CB_DROPDOWN | wx.CB_READONLY, value='Rising edge', choices=['Rising edge', 'Local loudness'])
        self.combo_kernel_type.SetToolTip(wx.ToolTip('Choose a kernel that responds to a rising edge or local loudness.'))
        self.label_magic_offset     = wx.StaticText(    self.panel_attack, label='Magic offset (ms): ±')
        self.spin_magic_offset      = wx.SpinCtrlDouble(self.panel_attack, style=wx.SP_ARROW_KEYS, min=-5.0, max=5.0, initial=2.0, inc=0.1)
        self.spin_magic_offset.SetToolTip(wx.ToolTip('Add a constant value to the time of maximum kernel response. I haven\'t tracked the cause of this down yet. Might be related to attack perception?'))
        self.spin_magic_offset.SetDigits(1)
        self.spin_magic_offset.Disable()

        # Sizing minimums
        for ctrl in [self.checkbox_null, self.checkbox_p9ms]:
            ctrl.SetMinSize((48, 24))
        for ctrl in [self.combo_kernel_target, self.combo_kernel_type]:
            ctrl.SetMinSize((108, 24))
        for spinner in [self.spin_fingerprint_size, self.spin_window_size, self.spin_step_size, self.spin_tolerance, self.spin_magic_offset]:
            spinner.SetMinSize((54, 24))
        
        sizer_attack = wx.GridBagSizer(6, 0)
        sizer_attack.Add(self.label_attack,        (0, 0), span=(1, 3), flag=wx.EXPAND | wx.ALIGN_CENTER)
        sizer_attack.Add(self.label_kernel_target, (1, 0),              flag=wx.EXPAND | wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
        sizer_attack.Add(self.combo_kernel_target, (1, 1), span=(1, 2), flag=wx.EXPAND | wx.CENTER)
        sizer_attack.Add(self.label_kernel_type,   (2, 0),              flag=wx.EXPAND | wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
        sizer_attack.Add(self.combo_kernel_type,   (2, 1), span=(1, 2), flag=wx.EXPAND | wx.CENTER)
        sizer_attack.Add(self.label_magic_offset,  (3, 0), span=(1, 2), flag=wx.EXPAND | wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
        sizer_attack.Add(self.spin_magic_offset,   (3, 2),              flag=wx.EXPAND | wx.CENTER)
        sizer_attack.AddGrowableCol(0, 1)
        sizer_v = wx.BoxSizer(wx.VERTICAL)
        sizer_h = wx.BoxSizer(wx.HORIZONTAL)
        sizer_h.Add(sizer_attack, 1, wx.CENTER)
        sizer_v.Add(sizer_h,      1, wx.CENTER)
        self.panel_attack.SetSizer(sizer_v)
        

        # --------------------------------------------------------------
        # Process!
        
        # October 11, 2006 (but I'm lazy about time zones)
        years_since_r21 = int((dt.utcnow() - dt(year=2006, month=10, day=12)) / td(days=365.2425))
        self.button_process = wx.Button(self.panel_main, label=f'did you know the ITG r21 patch is over {years_since_r21} years old? let that sync in')
        self.button_process.SetToolTip(wx.ToolTip('Process all simfiles in the directory above using the specified parameters.'))
        self.button_process.SetMinSize((528, 30))
        self.button_process.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False))
        self.button_process.SetBackgroundColour(wx.Colour(170, 255, 255))


        # --------------------------------------------------------------
        # Results and resync offers
        self.panel_results = wx.Panel(self.panel_main)
        self.panel_results.SetBackgroundColour(self.GetBackgroundColour().ChangeLightness(170))
        self.panel_results.SetMinSize((324, 90))
        
        self.label_results  = wx.StaticText(self.panel_results, label='Sync bias results')
        self.label_results.SetFont(self.label_results.GetFont().MakeUnderlined())
        self.label_logs     = wx.StaticText(self.panel_results, label='View logs: ')
        self.button_logs    = wx.BitmapButton(self.panel_results, bitmap=wx.ArtProvider.GetBitmap(wx.ART_REPORT_VIEW))
        self.button_logs.SetToolTip(wx.ToolTip('Open the log file for the run...'))
        self.label_plots    = wx.StaticText(self.panel_results, label='View plots: ')
        self.button_plots   = wx.BitmapButton(self.panel_results, bitmap=wx.ArtProvider.GetBitmap(wx.ART_FOLDER_OPEN))
        self.button_plots.SetToolTip(wx.ToolTip('Open the directory above that contains the plots...'))

        
        self.label_null     = wx.StaticText(self.panel_results, label=' Null (StepMania) ')
        self.label_p9ms     = wx.StaticText(self.panel_results, label=' +9ms (In The Groove) ')
        self.label_unknown  = wx.StaticText(self.panel_results, label=' Unknown paradigm: ')
        self.entry_null     = wx.TextCtrl(self.panel_results, value='----', style=wx.TE_CENTER)
        self.entry_p9ms     = wx.TextCtrl(self.panel_results, value='----', style=wx.TE_CENTER)
        self.entry_unknown  = wx.TextCtrl(self.panel_results, value='----', style=wx.TE_CENTER)
        self.button_null    = wx.BitmapButton(self.panel_results, bitmap=wx.ArtProvider.GetBitmap(wx.ART_GO_DOWN))
        self.button_null.SetToolTip(wx.ToolTip('Add a +9ms bias to these simfiles or charts.'))
        self.button_p9ms    = wx.BitmapButton(self.panel_results, bitmap=wx.ArtProvider.GetBitmap(wx.ART_GO_UP))
        self.button_p9ms.SetToolTip(wx.ToolTip('Remove the +9ms bias from these simfiles or charts.'))

        for button in [self.button_logs, self.button_plots, self.button_null, self.button_p9ms]:
            button.SetMinSize((28, 28))
        for entry in [self.entry_null, self.entry_p9ms, self.entry_unknown]:
            entry.SetMinSize((48, 24))
            entry.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False))

        sizer_null = wx.BoxSizer(wx.HORIZONTAL)
        sizer_null.Add(self.label_null,  1, wx.CENTER)
        sizer_null.Add(self.entry_null,  0, wx.CENTER)
        sizer_null.AddSpacer(6)
        sizer_null.Add(self.button_null, 0, wx.CENTER)
        sizer_p9ms = wx.BoxSizer(wx.HORIZONTAL)
        sizer_p9ms.Add(self.button_p9ms, 0, wx.CENTER)
        sizer_p9ms.AddSpacer(6)
        sizer_p9ms.Add(self.entry_p9ms,  0, wx.CENTER)
        sizer_p9ms.Add(self.label_p9ms,  1, wx.CENTER)
        sizer_unknown = wx.BoxSizer(wx.HORIZONTAL)
        sizer_unknown.Add(self.label_unknown, 1, wx.CENTER)
        sizer_unknown.Add(self.entry_unknown, 0, wx.CENTER)
        sizer_results = wx.GridBagSizer(0, 6)
        sizer_results.Add(self.label_results, (0, 0), span=(1, 2), flag=wx.EXPAND | wx.ALIGN_CENTER)
        sizer_results.Add(self.label_logs,    (1, 0),              flag=wx.EXPAND | wx.ALIGN_CENTER)
        sizer_results.Add(self.button_logs,   (1, 1),              flag=wx.EXPAND | wx.ALIGN_CENTER)
        sizer_results.Add(self.label_plots,   (2, 0),              flag=wx.EXPAND | wx.ALIGN_CENTER)
        sizer_results.Add(self.button_plots,  (2, 1),              flag=wx.EXPAND | wx.ALIGN_CENTER)
        sizer_results.Add(sizer_null,         (0, 3),              flag=wx.EXPAND | wx.ALIGN_CENTER)
        sizer_results.Add(sizer_p9ms,         (1, 3),              flag=wx.EXPAND | wx.ALIGN_CENTER)
        sizer_results.Add(sizer_unknown,      (2, 3),              flag=wx.EXPAND | wx.ALIGN_CENTER)
        sizer_results.AddGrowableCol(0, 1)
        sizer_results.AddGrowableCol(2, 1)
        sizer_v = wx.BoxSizer(wx.VERTICAL)
        sizer_h = wx.BoxSizer(wx.HORIZONTAL)
        sizer_h.Add(sizer_results, 1, wx.CENTER)
        sizer_v.Add(sizer_h,      1, wx.CENTER)
        self.panel_results.SetSizer(sizer_v)

        # --------------------------------------------------------------
        # Results table
        self.grid_results = wx.grid.Grid(self.panel_main)
        self.grid_results.CreateGrid(0, 4)
        self.grid_results.SetMinSize((318, 312))
        
        self.grid_results.DisableCellEditControl()
        self.grid_results.DisableDragColMove()
        self.grid_results.DisableDragColSize()
        self.grid_results.DisableDragRowMove()
        self.grid_results.DisableDragRowSize()
        self.grid_results.DisableHidingColumns()

        for row_index in range(self.grid_results.GetNumberRows()):
            for col_index in range(self.grid_results.GetNumberCols()):
                self.grid_results.SetReadOnly(row_index, col_index)

        self.grid_results.HideRowLabels()
        self.grid_results.SetColLabelSize(18)
        for row_index in range(self.grid_results.GetNumberRows()):
            self.grid_results.SetRowSize(row_index, 18)
        for col_index, col_label in enumerate(['Simfile', 'Slot', 'Bias', 'Par?']):
            self.grid_results.SetColLabelValue(col_index, col_label)
        for col_index, col_width in enumerate([180, 36, 48, 36]):
            self.grid_results.SetColSize(col_index, col_width)
        self.grid_results.SetDefaultCellAlignment(wx.ALIGN_CENTER, wx.ALIGN_CENTER)
        self.grid_results.SetDefaultCellFitMode(wx.grid.GridFitMode.Ellipsize())



        # --------------------------------------------------------------
        # Plots
        self.panel_plot = wx.Panel(self.panel_main)
        self.panel_plot.figure = Figure(dpi=48)
        gs = self.panel_plot.figure.add_gridspec(3)         # hspace
        self.panel_plot.axes = gs.subplots(sharex=True, sharey=False)
        self.panel_plot.figure.suptitle('Sync fingerprint\nArtist - "Title"\nSync bias: +0.003 (probably null)')
        x_test = np.linspace(-50, 50, 101, endpoint=True)
        # self.panel_plot.axes[0].plot(x_test, np.sin(x_test / 10), 'r-')
        # self.panel_plot.axes[1].plot(x_test, np.sin(x_test / 12), 'g-')
        # self.panel_plot.axes[2].plot(x_test, np.sin(x_test / 15), 'b-')
        self.panel_plot.axes[0].set_ylabel('Frequency [kHz]')
        self.panel_plot.axes[1].set_ylabel('Beat index')
        self.panel_plot.axes[2].set_ylabel('Post-kernel fingerprint')
        self.panel_plot.axes[2].set_xlabel('Offset from beat [ms]')
        for i in range(3):
            self.panel_plot.axes[i].set_box_aspect(2/3)
            self.panel_plot.axes[i].margins(x=0.01, y=0.01)
            self.panel_plot.axes[i].set_yticks([])
        self.panel_plot.figure.subplots_adjust(
            left=0.08,
            right=0.99,
            bottom=0.05,
            top=0.89,
            wspace=0,
            hspace=0
        )
        self.panel_plot.canvas = FigureCanvas(self.panel_main, -1, self.panel_plot.figure)
        self.panel_plot.canvas.SetMinSize((180, 402))
        self.panel_plot.canvas.SetToolTip(wx.ToolTip('Double-click on a result row to examine the audio fingerprint.\n(These plots are also stored in the report directory)'))
        
        # --------------------------------------------------------------
        # Menu and status bar
        self.menu_setup()

        self.CreateStatusBar()
        self.SetStatusText('hi')


        # --------------------------------------------------------------
        # Bindings
        self.bindings_setup()


        # --------------------------------------------------------------
        # Layout
        sizer_parameters = wx.GridBagSizer(6, 6)
        sizer_parameters.Add(self.panel_paradigms,   (0, 0), flag=wx.CENTER)
        sizer_parameters.Add(self.panel_fingerprint, (0, 1), flag=wx.CENTER)
        sizer_parameters.Add(self.panel_attack,      (0, 2), flag=wx.CENTER)

        sizer_lower_half = wx.BoxSizer(wx.HORIZONTAL)
        sizer_results_pane = wx.BoxSizer(wx.VERTICAL)
        sizer_results_pane.Add(self.panel_results, 0, wx.CENTER)
        # sizer_results_pane.AddSpacer(6)
        sizer_results_pane.Add(self.grid_results, 0, wx.CENTER)
        sizer_lower_half.Add(sizer_results_pane, 0, wx.CENTER)
        sizer_lower_half.AddSpacer(6)
        sizer_lower_half.Add(self.panel_plot.canvas, 0, wx.CENTER)

        sizer_all = wx.BoxSizer(wx.VERTICAL)
        sizer_all.Add(sizer_paths, 0, wx.CENTER)
        sizer_all.AddSpacer(6)
        sizer_all.Add(sizer_parameters, 0, wx.CENTER)
        sizer_all.AddSpacer(6)
        sizer_all.Add(self.button_process, 0, wx.CENTER)
        sizer_all.AddSpacer(6)
        sizer_all.Add(sizer_lower_half, 0, wx.CENTER)

        sizer_v = wx.BoxSizer(wx.VERTICAL)
        sizer_h = wx.BoxSizer(wx.HORIZONTAL)
        sizer_h.Add(sizer_all, 1, wx.CENTER)
        sizer_v.Add(sizer_h,   1, wx.CENTER)
        self.panel_main.SetSizer(sizer_v)
        self.panel_main.Layout()


    def menu_setup(self):
        self.menu_file = wx.Menu()
        self.menu_item_exit = self.menu_file.Append(wx.ID_EXIT)

        self.menu_help = wx.Menu()
        self.menu_item_doc = self.menu_help.Append(wx.ID_HELP, helpString='Navigate to online documentation at GitHub')
        self.menu_item_about = self.menu_help.Append(wx.ID_ABOUT)

        self.menu_bar = wx.MenuBar()
        self.menu_bar.Append(self.menu_file, '&File')
        self.menu_bar.Append(self.menu_help, '&Help')
        self.SetMenuBar(self.menu_bar)

        self.Bind(wx.EVT_MENU, self.OnExit,  self.menu_item_exit)
        self.Bind(wx.EVT_MENU, self.OnAbout, self.menu_item_about)
        self.Bind(wx.EVT_MENU, self.OnHelp,  self.menu_item_doc)

    def bindings_setup(self):
        self.Bind(wx.EVT_TOGGLEBUTTON, self.OnTogglePathAuto, self.button_report_path_auto)
        self.Bind(wx.EVT_BUTTON, self.OnChooseRootPath, self.button_root)
        self.Bind(wx.EVT_BUTTON, self.OnChooseReportPath, self.button_report_path)
        self.Bind(wx.EVT_BUTTON, self.OnProcess, self.button_process)
        self.Bind(wx.EVT_BUTTON, self.OnOpenLogs, self.button_logs)
        self.Bind(wx.EVT_BUTTON, self.OnViewPlots, self.button_plots)
        self.Bind(wx.grid.EVT_GRID_CELL_LEFT_DCLICK, self.OnClickResultRow, self.grid_results)
        self.Bind(wx.EVT_BUTTON, self.OnConvert9msToNull, self.button_p9ms)
        self.Bind(wx.EVT_BUTTON, self.OnConvertNullTo9ms, self.button_null)


    def collect_parameters(self):
        params = {}
        params['root_path'] = self.entry_root.GetValue()
        params['report_path'] = self.entry_report_path.GetValue()
        params['consider_null'] = self.checkbox_null.IsChecked()
        params['consider_p9ms'] = self.checkbox_p9ms.IsChecked()
        params['tolerance'] = self.spin_tolerance.GetValue()
        params['fingerprint_ms'] = self.spin_fingerprint_size.GetValue()
        params['window_ms'] = self.spin_window_size.GetValue()
        params['step_ms'] = self.spin_step_size.GetValue()
        params['kernel_target'] = self.combo_kernel_target.GetSelection()
        params['kernel_type'] = self.combo_kernel_type.GetSelection()
        params['magic_offset'] = self.spin_magic_offset.GetValue()
        return params
    
    def allow_to_update(self):
        wx.Yield()


    def OnTogglePathAuto(self, event):
        if event.IsChecked():
            self.entry_report_path.Disable()
            self.button_report_path.Disable()
            self.entry_report_path.SetValue(os.path.join(self.entry_root.GetValue(), '__bias-check'))
        else:
            self.entry_report_path.Enable()
            self.button_report_path.Enable()

    def OnChooseRootPath(self, event):
        dlg = wx.DirDialog(self,
            message='Choose a simfile, pack, or group of packs:',
            style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST
        )
        if dlg.ShowModal() == wx.ID_OK:
            self.entry_root.SetValue(dlg.GetPath())
            if self.button_report_path_auto.GetValue():
                self.entry_report_path.SetValue(os.path.join(dlg.GetPath(), '__bias-check'))
        dlg.Destroy()

    def OnChooseReportPath(self, event):
        dlg = wx.DirDialog(self,
            message='Choose a directory for the sync bias report and audio fingerprint plots:',
            style=wx.DD_DEFAULT_STYLE,
            defaultPath=os.path.join(self.entry_root.GetValue(), '__bias-check')
        )
        if dlg.ShowModal() == wx.ID_OK:
            self.entry_report_path.SetValue(dlg.GetPath())
        dlg.Destroy()

    def OnProcess(self, event):
        print('Process')

        params = self.collect_parameters()

        # Verify existence of root path
        if not os.path.isdir(params['root_path']):
            wx.MessageBox('Root directory doesn\'t exist\n\nDouble-check the paths in the text boxes at the top of this window.', caption='+9ms or Null?', style=wx.ICON_ERROR)
            event.Skip()
            return
        else:
            print(f"Root directory exists: {params['root_path']}")

        # Verify existence of root path
        if not os.path.isdir(params['report_path']):
            try:
                os.makedirs(params['report_path'])
                print(f"Report directory created: {params['report_path']}")
            except Exception as e:
                wx.MessageBox('Report directory can\'t be created\n\nDouble-check the paths in the text boxes at the top of this window.', caption='+9ms or Null?', style=wx.ICON_ERROR)
                event.Skip()
                return
        else:
            print(f"Report directory exists: {params['report_path']}")

        # Set up logging
        logging.getLogger().handlers.clear()

        log_path = os.path.join(params['report_path'], 'nine-or-null.log')
        log_fmt = logging.Formatter(
            '[%(asctime)s.%(msecs)03d] %(levelname)-8s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.basicConfig(
            filename=log_path,
            encoding='utf-8',
            level=logging.INFO
        )
        logging.getLogger().addHandler(logging.StreamHandler())
        for handler in logging.getLogger().handlers:
            handler.setFormatter(log_fmt)

        csv_path = os.path.join(params['report_path'], 'nine-or-null.csv')

        # Recall parameters.
        header_str = f'+9ms or Null? v{_VERSION} (GUI)'
        logging.info(f"{'=' * 20}{header_str:^32s}{'=' * 20}")
        logging.info('Parameter settings:')
        for k, v in params.items():
            logging.info(f'\t{k} = {v}')

        # Clear grid and counters.
        if self.grid_results.GetNumberRows() > 0:
            self.grid_results.DeleteRows(0, self.grid_results.GetNumberRows())
        self.entry_null.SetValue(   '----')
        self.entry_p9ms.SetValue(   '----')
        self.entry_unknown.SetValue('----')

        with open(csv_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=_CSV_FIELDNAMES, extrasaction='ignore')
            writer.writeheader()

            self.fingerprints = batch_process(
                plot_hook_gui=self.panel_plot,
                grid_hook_gui=self.grid_results,
                gui_hook=self,
                csv_hook=writer,
                **params
            )
        self.SetStatusText(f'Done! {len(self.fingerprints)} fingerprints processed.')

        logging.info('-' * 72)
        logging.info(f"Sync bias report: {len(self.fingerprints)} fingerprints processed in {params['root_path']}")

        paradigm_count = {}
        for paradigm in ['+9ms', 'null', '????']:
            paradigm_map = {k: v for k, v in self.fingerprints.items() if guess_paradigm(v['bias_result']) == paradigm}
            logging.info(f"Files sync'd to {paradigm}: {len(paradigm_map)}")
            for k, v in paradigm_map.items():
                logging.info(f"\t{k:>50s}")
                logging.info(f"\t\tderived sync bias = {v['bias_result']:+0.1f} ms")
            paradigm_count[paradigm] = len(paradigm_map)

        if params['consider_null']:
            self.entry_null.SetValue(f"{paradigm_count['null']}")
        if params['consider_p9ms']:
            self.entry_p9ms.SetValue(f"{paradigm_count['+9ms']}")
        self.entry_unknown.SetValue( f"{paradigm_count['????']}")

        paradigm_most = sorted([k for k in paradigm_count], key=lambda k: paradigm_count.get(k, 0))
        logging.info('=' * 72)
        logging.info(f'Pack sync paradigm: {paradigm_most[-1]}')
        logging.info('-' * 72)

        # Done!


    def OnOpenLogs(self, event):
        print('View Logs')
        log_paths = [
            handler.baseFilename for handler in logging.getLogger().handlers if isinstance(handler, logging.FileHandler)
        ]
        if len(log_paths) > 0 and os.path.isfile(log_paths[0]):
            wx.LaunchDefaultApplication(log_paths[0])
        else:
            wx.MessageBox('No logs found\n\nYou may not have run a sync bias analysis yet.', caption='+9ms or Null?', style=wx.ICON_ERROR)

    def OnViewPlots(self, event):
        print('Open Plots Directory')
        report_path = self.entry_report_path.GetValue()
        if os.path.isdir(report_path):
            wx.LaunchDefaultBrowser(report_path)
        else:
            wx.MessageBox('Report directory not found\n\nYou may not have run a sync bias analysis yet.', caption='+9ms or Null?', style=wx.ICON_ERROR)

    def OnClickResultRow(self, event):
        print('Double-clicked Result Row')
        params = self.collect_parameters()
        row_to_key = [k for k in self.fingerprints.keys()]
        # print([f'{i}: {k}' for i, k in enumerate(row_to_key)])
        plot_fingerprint(self.fingerprints[row_to_key[event.GetRow()]], self.panel_plot.figure.get_axes(), hide_yticks=True, **params)
        self.panel_plot.canvas.draw()
        event.Skip()

    def OnConvert9msToNull(self, event):
        wx.MessageBox('Coming soon... ;)\n\nConvert +9ms (In The Groove) to null (StepMania)', caption='+9ms or Null?')

    def OnConvertNullTo9ms(self, event):
        wx.MessageBox('Coming soon... ;)\n\nConvert null (StepMania) to +9ms (In The Groove)', caption='+9ms or Null?')
    
    def OnExit(self, event):
        self.Close(True)

    def OnHelp(self, event):
        wx.LaunchDefaultBrowser('https://github.com/telperion/nine-or-null')

    def OnAbout(self, event):
        dlg = AboutWithLinks(self, wx.ID_ABOUT, title=f'About +9ms or Null? v{_VERSION}')
        dlg.ShowModal()


def start_gui():
    app = wx.App()
    frame = NineOrNull(None, title=f'+9ms or Null? v{_VERSION}', style=wx.DEFAULT_FRAME_STYLE & ~(wx.RESIZE_BORDER | wx.MAXIMIZE_BOX))
    frame.Show()
    app.MainLoop()


if __name__ == '__main__':
    start_gui()
