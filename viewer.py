import tkinter
from tkinter import filedialog as tkFileDialog
import time
import threading
import queue as Queue
import cv2
from PIL import Image, ImageTk
import numpy as np
from datetime import datetime
import signal
import sys
import jsonpickle
import message # used for unpickling json str to message obj

from subscriber import Subscriber
from drawables import Trajectory, ScannerData, Landmarks, Points, Particles


class GuiPart:
    def __init__(self, root, queue, endCommand):
        self.queue = queue
        self.root = root

        self.window_extents = (1200, 1000)
        # The extents of the world and sensor canvas.
        self.world_canvas_extents = (400, (self.window_extents[1]- 1)/2)
        self.sensor_canvas_extents = (400, (self.window_extents[1]- 1)/2)
        self.camera_canvas_extents = (848, (self.window_extents[1]- 1)/2)
        self.log_canvas_extents = (400, (self.window_extents[1]- 1)/2)

        # camera image size
        self.camera_img_size = (848, 480)

        # The world extents of the scene in meters
        self.world_extents = (8.0, 8.0) # world extents, in meters
        self.scanner_range = 2.0 # range of camera, in meters

        root.geometry(str(self.window_extents[0]) + "x" + str(self.window_extents[1]))
        root.title('Robot Viewer')

        # Setup GUI stuff.
        frame1 = tkinter.Frame(root)
        frame1.pack(fill=tkinter.BOTH, expand=True)
        self.world_canvas = tkinter.Canvas(frame1, height=self.world_canvas_extents[1],bg="white")
        self.world_canvas.pack(side=tkinter.LEFT, anchor=tkinter.NW, fill=tkinter.BOTH, expand=True, pady=2, padx=2)
        self.sensor_canvas = tkinter.Canvas(frame1, height=self.sensor_canvas_extents[1],bg="white")
        self.sensor_canvas.pack(side=tkinter.RIGHT, anchor=tkinter.NE, fill=tkinter.BOTH, expand=True, pady=2, padx=2)

        self.frame2 = tkinter.Frame(root)
        self.frame2.pack(fill=tkinter.BOTH, expand=True)
        # self.scale = tkinter.Scale(self.frame2, orient=tkinter.HORIZONTAL, command =self.slider_moved)
        # self.scale.pack(fill=tkinter.X)
        # self.info = tkinter.Label(self.frame2)
        # self.info.pack()
        # load = tkinter.Button(self.frame2,text="Load (additional) logfile",command=self.add_file)
        # load.pack(side=tkinter.LEFT)
        # reload_all = tkinter.Button(self.frame2,text="Reload all",command=self.load_data)
        # reload_all.pack(side=tkinter.RIGHT)
        # done_btn = tkinter.Button(self.frame2, text='Done', command=endCommand)
        # done_btn.pack()

        frame3 = tkinter.Frame(root)
        frame3.pack(fill=tkinter.BOTH, expand=True)
        self.camera_canvas = tkinter.Canvas(frame3,width=self.camera_canvas_extents[0], height=self.camera_canvas_extents[1],bg="gray")
        self.camera_canvas.pack(side=tkinter.LEFT, anchor=tkinter.NW, fill=tkinter.BOTH, expand=True, pady=2, padx=2)

        self.log_canvas = tkinter.Canvas(frame3, width=self.log_canvas_extents[0], height=self.log_canvas_extents[1],bg="gray")
        self.log_canvas.pack(side=tkinter.RIGHT, anchor=tkinter.NE, fill=tkinter.BOTH, expand=True, pady=2, padx=2)

        # The list of objects to draw.
        self.draw_objects = []

        # Ask for file.
        self.all_file_names = []
        # self.add_file()
        root.update()
        root.update_idletasks() 

        self.width, self.height = self.root.winfo_width(), self.root.winfo_height()
        root.bind("<Configure>", self.resize_event)


    def resize_event(self, event):
        if(event.widget == self.root and (self.width != event.width or self.height != event.height)
            and (event.width != 0 or event.height != 0)):

            self.width, self.height = event.width, event.height
            self.resize()


    def resize(self):
        fixed_height_part = self.frame2.winfo_height() + 10 # add 10 to make sure none of the image is cropped

        max_canvas_width = (self.width)/2
        max_canvas_height = (self.height - fixed_height_part)/2
        
        self.world_canvas_extents = (max_canvas_width, max_canvas_height)
        self.sensor_canvas_extents = (max_canvas_width, max_canvas_height)

        self.world_canvas.config(width=self.world_canvas_extents[0], height=self.world_canvas_extents[1])
        self.sensor_canvas.config(width=self.sensor_canvas_extents[0], height=self.sensor_canvas_extents[1])

        camera_img_ratio = self.camera_img_size[0]/self.camera_img_size[1]
        # print("camera_img_ratio", camera_img_ratio)

        self.camera_canvas_extents = (int(max_canvas_height * camera_img_ratio), int(max_canvas_height+2))
        self.camera_canvas.config(width=self.camera_canvas_extents[0], height=self.camera_canvas_extents[1])

        # print("camera_canvas_extents ratio", self.camera_canvas_extents[0]/self.camera_canvas_extents[1])
        # print("camera_canvas_extents ratio", self.camera_canvas.winfo_width()/self.camera_canvas.winfo_height())

        self.log_canvas_extents = (self.width - self.camera_canvas_extents[0],  self.camera_canvas_extents[1])
        self.log_canvas.config(width=self.log_canvas_extents[0], height=self.log_canvas_extents[1])

    def slider_moved(self, index):
        """Callback for moving the scale slider."""
        i = int(index)
        # Call all draw objects.
        # for d in draw_objects:
        #     d.draw(i)

    def add_file(self):
        filename = tkFileDialog.askopenfilename(filetypes = [("all files", ".*"), ("txt files", ".txt")])
        if filename:
            # If the file is in the list already, remove it (so it will be appended
            # at the end).
            if filename in self.all_file_names:
                self.all_file_names.remove(filename)
            self.all_file_names.append(filename)
            self.load_data()

    def to_sensor_canvas(self, r, alpha):
        """Transforms a point from sensor coordinates (meters) to sensor canvas coord system."""
        # convert (r, alpha) to (x, y)
        rel_x = np.cos(alpha) * r
        rel_y = np.sin(alpha) * r

        # divide canvas_extents by 2 because we want to have the camera in the middle of the canvas
        scale = (1/self.scanner_range) * (np.array(self.sensor_canvas_extents)/2)

        # x and y swapped on purpose
        x = self.sensor_canvas_extents[0] / 2.0 - rel_y * scale[0]
        x = x.astype(int)
        y = self.sensor_canvas_extents[1] / 2.0 - 1 - rel_x * scale[1]
        y = y.astype(int)

        points = np.column_stack((x, y))
        return points

    def to_world_canvas(self, world_x, world_y):
        """Transforms a point from world coord system to world canvas coord system."""

        scale = (1/np.array(self.world_extents)) * np.array(self.world_canvas_extents)
        x = np.array(world_x) * scale[0] + self.world_canvas_extents[0]/2
        x = x.astype(int)
        y = self.world_canvas_extents[1]/2 - np.array(world_y) * scale[1]
        y = y.astype(int)
        points = np.column_stack((x, y))
        return list(points)

    def load_new_data(self, msg):

        draw_objects = []

        # draw axes
        draw_objects.append(ScannerData(self.sensor_canvas,
            self.sensor_canvas_extents, self.scanner_range))

        landmark_points = self.to_sensor_canvas(msg.landmark_rs, msg.landmark_alphas)

        landmark_points = np.array([landmark_points])
        landmark_ids = np.array([msg.landmark_ids])
        landmark_stdevs = list([msg.landmark_stdevs])

        if len(landmark_points[0]) > 0:
            draw_objects.append(Points(landmark_points, self.sensor_canvas, "#cc0000", ids=landmark_ids))

        # robot world coordinate:
        robot_world_points = self.to_world_canvas(msg.x, msg.y)
        robot_stdev = list([msg.stdev])

        if len(robot_world_points[0]) > 0:
            # black points
            # draw_objects.append(Points(robot_world_points, self.world_canvas, "#262339"))

            # red points with ellipse
            draw_objects.append(Trajectory(robot_world_points, self.world_canvas, self.world_extents, self.world_canvas_extents,
                standard_deviations = robot_stdev,
                cursor_color="blue", background_color="lightblue",
                position_stddev_color = "#8080ff", theta_stddev_color="#c0c0ff"))

        # landmark world coodinates:
        landmark_world_points = self.to_world_canvas(msg.landmark_xs, msg.landmark_ys)
        landmark_world_points = np.array([landmark_world_points])

        if len(landmark_world_points[0]) > 0:
            # black points
            draw_objects.append(Points(landmark_world_points, self.world_canvas, "#262339", ids=landmark_ids))

            # landmark world ellipses and red points
            factor = self.world_canvas_extents[0] / self.world_extents[0]
            draw_objects.append(Points(landmark_world_points, self.world_canvas, "#cc0000",
                                    ellipses=landmark_stdevs,
                                    ellipse_factor=factor))

        # Start new canvas and do all background drawing.
        self.world_canvas.delete(tkinter.ALL)
        self.sensor_canvas.delete(tkinter.ALL)
        # draw axes
        for d in draw_objects:
            d.background_draw()

        # Call all draw objects.
        for d in draw_objects:
            d.draw(0)



    def processIncoming(self):
        """Handle all messages currently in the queue, if any."""
        while self.queue.qsize():
            try:
                msg_str, img = self.queue.get(0)
                # Check contents of message and do whatever is needed. As a
                # simple test, print it (in real life, you would
                # suitably update the GUI's display in a richer fashion).
                msg = jsonpickle.decode(msg_str)
                date_time = datetime.fromtimestamp(msg.timestamp)
                date_time_str = date_time.strftime("%m/%d/%Y, %H:%M:%S.%f")

                cv2image= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                self.camera_img_size = img.size
                img = img.resize((int(self.camera_canvas_extents[0]), int(self.camera_canvas_extents[1])), Image.ANTIALIAS)

                # Convert image to PhotoImage
                self.imgtk = ImageTk.PhotoImage(image = img)
                self.camera_canvas.delete("IMG")
                self.camera_canvas.create_image(0, 0, image=self.imgtk, anchor=tkinter.NW, tags="IMG")
                
                text = date_time_str + "\n"
                text += "img size: " + str(self.camera_img_size) + "\n"
                text += "id: " + str(msg.id) + "\n"
                text += "start: " + str(msg.start) + "\n"
                text += "landmark ids: " + str(msg.landmark_ids) + "\n"
                text += "robot pos: (" + str(np.round(msg.x, 2)) + ", " + str(np.round(msg.y, 2)) + ")\n"
                self.log_canvas.delete("TEXT")
                self.log_canvas.create_text(10,10,fill="black", anchor=tkinter.NW,
                        text=text, tags="TEXT")

                if msg:
                    self.load_new_data(msg)

            except Queue.Empty:
                # just on general principles, although we don't
                # expect this branch to be taken in this case
                pass


class ThreadedClient:
    """
    Launch the main part of the GUI and the worker thread. periodicCall and
    endApplication could reside in the GUI part, but putting them here
    means that you have all the thread controls in a single place.
    """
    def __init__(self, root):
        """
        Start the GUI and the asynchronous threads. We are in the main
        (original) thread of the application, which will later be used by
        the GUI as well. We spawn a new thread for the worker (I/O).
        """
        self.root = root

        # Create the queue
        self.queue = Queue.Queue()

        # Set up the GUI part
        self.gui = GuiPart(root, self.queue, self.endApplication)

        self.subscriber = Subscriber()

        # Set up the thread to do asynchronous I/O
        # More threads can also be created and used, if necessary
        self.running = 1
        # self.thread1 = threading.Thread(target=self.workerThread1)

        # Start the periodic call in the GUI to check if the queue contains
        # anything
        self.periodicCall()

        print("starting thread...")
        self.thread1 = threading.Thread(target=self.workerThread1)
        self.thread1.start()


    def periodicCall(self):
        """
        Check every 200 ms if there is something new in the queue.
        """
        self.gui.processIncoming()
        if not self.running:
            # This is the brutal stop of the system. You may want to do
            # some cleanup before actually shutting it down.
            self.subscriber.close()

            print("exiting...")
            sys.exit(1)

        self.root.after(10, self.periodicCall)

    def workerThread1(self):
        """
        This is where we handle the asynchronous I/O. For example, it may be
        a 'select(  )'. One important thing to remember is that the thread has
        to yield control pretty regularly, by select or otherwise.
        """
        while self.running:
            # Asynchronous stuff here 
            msg_str, img = self.subscriber.run()
            self.queue.put((msg_str, img))

    def endApplication(self):
        self.running = 0

if __name__ == '__main__':
    client = None
    root = tkinter.Tk()
    # root.tk.call('tk', 'scaling', 2.0) # doesn't do anything

    client = ThreadedClient(root)

    # handle ctrl + c
    def sigint_handler(sig, frame):
        print("called end application...")
        client.endApplication()

    signal.signal(signal.SIGINT, sigint_handler)

    root.mainloop()        