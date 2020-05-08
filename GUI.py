#-*- coding:utf-8 -*-
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
import sys
from tkinter import ttk
from tkinter.filedialog import askdirectory
from pathlib import Path
import os
import pynvml



class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master,bg='black')
        self.font_name = '微软雅黑'
        self.font_size = 16
        self.padx = 5
        self.pady = 5
        self.test_scripts_root = "./test_scripts"
        self.train_logs_root = "./train_logs"
        self.test_logs_root = "./test_logs"
        self.test_img_root = "D:/PatchFace/PleaseWork/Benchmark/styletransfer"
        self.window_init()
        self.createWidgets()
        

    
    def window_init(self):
        self.master.title('Test System')
        # self.master.bg='black'
        self.master.iconbitmap('./cells.ico')
        width,height=self.master.maxsize()
        self.master.geometry("{}x{}".format(600, height//2))
        self.master.resizable(0,0)
       
    def createWidgets(self):
        font_list = (self.font_name,self.font_size)
        i = 0
        self.mode_label = tk.Label(self.master,text="Mode",font=font_list,fg = "black")
        self.mode_label.grid(row=i,padx=self.padx,pady=self.pady)
        self.mode_str = tk.StringVar()
        self.mode_Chosen = ttk.Combobox(self.master,font=font_list, width=30, textvariable=self.mode_str)
        self.mode_Chosen['values'] = ("Training Mode", "Finetune Mode", "Test Mode")
        self.mode_Chosen.grid(row=i,column=1,padx=self.padx,pady=self.pady)
        self.mode_Chosen.current(2)

        child_paths = []
        parent_path = Path(self.train_logs_root)
        child_path_iter = parent_path.iterdir()
        for child_path in child_path_iter:
            child_paths.append(child_path.name)
        i += 1
        self.version_label = tk.Label(self.master,text="Version",font=font_list,fg = "black")
        self.version_label.grid(row=i,padx=self.padx,pady=self.pady)
        self.version_str = tk.StringVar()
        self.version_Chosen = ttk.Combobox(self.master,font=font_list, width=30, textvariable=self.version_str)
        self.version_Chosen['values'] = child_paths
        self.version_Chosen.grid(row=i,column=1,padx=self.padx,pady=self.pady)
        self.version_Chosen.current(0)

        i += 1
        self.ckpt_label = tk.Label(self.master,text="Checkpoint",font=font_list,fg = "black")
        self.ckpt_label.grid(row=i,padx=self.padx,pady=self.pady)
        self.ckpt_str = tk.StringVar()
        self.ckpt_str.set("2000")
        self.ckpt_entry = tk.Entry(self.master,font=font_list, width=30, textvariable=self.ckpt_str)
        self.ckpt_entry.grid(row=i,column=1,padx=self.padx,pady=self.pady)

        i += 1
        self.node_label = tk.Label(self.master,text="Node Name",font=font_list,fg = "black")
        self.node_label.grid(row=i,padx=self.padx,pady=self.pady)
        self.node_str = tk.StringVar()
        self.node_Chosen = ttk.Combobox(self.master,font=font_list, width=30, textvariable=self.node_str)
        self.node_Chosen['values'] = ("localhost", "4card", "8card")
        self.node_Chosen.grid(row=i,column=1,padx=self.padx,pady=self.pady)
        self.node_Chosen.current(0)

        pynvml.nvmlInit()
        gpu_num = pynvml.nvmlDeviceGetCount()
        device_list = ["CPU"]
        for gpu_i in range(gpu_num):
            device_list.append("GPU:%d"%gpu_i)
        i += 1
        self.gpu_label = tk.Label(self.master,text="Device Name",font=font_list,fg = "black")
        self.gpu_label.grid(row=i,padx=self.padx,pady=self.pady)
        self.gpu_str = tk.StringVar()
        self.gpu_Chosen = ttk.Combobox(self.master,font=font_list, width=30, textvariable=self.gpu_str)
        self.gpu_Chosen['values'] = device_list
        self.gpu_Chosen.grid(row=i,column=1,padx=self.padx,pady=self.pady)
        self.gpu_Chosen.current(1)

        test_scripts = Path(self.test_scripts_root).glob('*.py') # ./*
        test_script_list = []
        for test_script_item in test_scripts:
            test_script_list.append(test_script_item.name[:-3])
        i += 1
        self.testScript_label = tk.Label(self.master,text="Test Script Name",font=font_list,fg = "black")
        self.testScript_label.grid(row=i,padx=self.padx,pady=self.pady)
        self.testScript_str = tk.StringVar()
        self.testScript_Chosen = ttk.Combobox(self.master,font=font_list, width=30, textvariable=self.testScript_str)
        self.testScript_Chosen['values'] = test_script_list
        self.testScript_Chosen.grid(row=i,column=1,padx=self.padx,pady=self.pady)
        self.testScript_Chosen.current(0)

        i += 1
        self.test_bs_label = tk.Label(self.master,text="Test Batch Size",font=font_list,fg = "black")
        self.test_bs_label.grid(row=i,padx=self.padx,pady=self.pady)
        self.test_bs_str = tk.StringVar()
        self.test_bs_scale = tk.Scale(self.master, from_=1, to=16, orient=tk.HORIZONTAL, length=360, showvalue=1,tickinterval=1, resolution=1)
        self.test_bs_scale.grid(row=i,column=1,padx=self.padx,pady=self.pady)

        i += 1
        self.test_img_path_label = tk.Label(self.master,text="Test Image Path",font=font_list,fg = "black")
        self.test_img_path_label.grid(row=i,padx=self.padx,pady=self.pady)
        self.test_img_path_str = tk.StringVar()
        self.test_img_path_str.set(self.test_img_root)
        fm1 = tk.Frame(self.master,width=30)
        fm1.grid(row=i,column=1,padx=self.padx,pady=self.pady)
        self.test_img_path_text = tk.Entry(fm1,width=28,font=font_list, textvariable = self.test_img_path_str)
        self.test_img_path_text.pack(side=tk.LEFT)

        self.test_img_path_button = tk.Button(fm1, text = "...", command = self.selectPath)
        self.test_img_path_button.pack(side=tk.RIGHT)

        i += 1
        self.logs_path_label = tk.Label(self.master,text="Logs Path",font=font_list,fg = "black")
        self.logs_path_label.grid(row=i,padx=self.padx,pady=self.pady)
        self.logs_path_str = tk.StringVar()
        self.logs_path_str.set(self.test_logs_root)
        fm2 = tk.Frame(self.master,width=30)
        fm2.grid(row=i,column=1,padx=self.padx,pady=self.pady)
        self.logs_path_text = tk.Entry(fm2,width=28,font=font_list, textvariable = self.logs_path_str)
        self.logs_path_text.pack(side=tk.LEFT)

        self.logs_path_button = tk.Button(fm2, text = "...", command = self.selectLogsPath)
        self.logs_path_button.pack(side=tk.RIGHT)

        i += 1
        self.run_test_button = tk.Button(self.master, width=30,text = "Run the test mode",font=font_list, command = self.runTest,bg='#006400', fg='#FF0000')
        self.run_test_button.grid(row=i, columnspan=2, padx=self.padx, pady=self.pady)

    def selectPath(self):
        path_ = askdirectory()
        if path_ !='':
            self.test_img_path_str.set(path_)
    
    def selectLogsPath(self):
        path_ = askdirectory()
        if path_ !='':
            self.logs_path_str.set(path_)


    def runTest(self):
        cwd = Path.cwd()
        mode_str = self.mode_str.get()
        if mode_str == "Training Mode":
            mode_str = "train"
        elif mode_str == "Finetune Mode":
            mode_str = "finetune"
        else:
            mode_str = "test"
        gpu_str = self.gpu_str.get()
        if gpu_str == "CPU":
            gpu_str = "-1"
        else:
            gpu_str = gpu_str[4:]
        version_str = self.version_str.get()
        node_str    = self.node_str.get()
        testScript_str = self.testScript_str.get()[7:]
        checkpoint_str = self.ckpt_str.get()
        img_root_str = self.test_img_path_str.get()
        cmd = "start cmd /k \"cd /d %s && conda activate pytorch11 && python main.py --mode %s --cuda %s --version %s --nodeName %s --testScriptName %s --checkpoint %s --testImgRoot %s"%(cwd,mode_str,gpu_str,version_str,node_str,testScript_str,checkpoint_str,img_root_str)
        os.system(cmd)

    def stop_program(self):
        self.thread._stop()
        sys.exit(0)

if __name__=='__main__':
    app = Application()
    # to do
    app.mainloop()
