# MATPLOTLIB 

""" 
Matplotlib is a data visualisation package for python

Very useful/important in lots of data science processes as a lot of these steps
involve data visaulisation.

So we usually get data, explore, train and model and then evaluate using
test data. 

We don't really want to have to look at numbers but rather box plots, histograms and all
the other data visualisation techniques to see patterns immediately.
"""

# import matplotlib for jupyter notebook
# installing matplotlib - pip3 install matplotlib for terminal

import numpy as np

""" 
Now we are going to use numpy aswell today

Going to use it to work efficiently with arrays, for taking data and feeding it into ML algorithms etc.

Numpy isn't a prerequisite for using matplotlib as you can use lists in matplotlib,
but because this is a data science society/tutorial sessions we're going to use numpy.
"""

import matplotlib.pyplot as plt

""" 
So firstly we're going to need some data that we want to visualise, and
we're going to use numpy to generate that data.

Below we're accessing random module in numpy and using the random function
to generate 50 values from, 0-1 and then we're multiplying it by 100 to get data points between
0 and 100
"""
X_data = np.random.random(50) * 100
Y_data = np.random.random(50) * 100

# Scatter plot

""" 
A bunch of different data points plotted as dots
"""

plt.Scatter(X_data, Y_data)     # creates a scatter plot

""" 
plt.show() in vscode or any other editor whereas jupyter notebook will 
present it/show the result straight after plt.scatter() etc.

Every time we run the code above we will get different data because the points are 
randomly generated
"""

plt.Scatter(X_data, Y_data, c="red")    # will give us red points
                                        # you can also use hex colours aswell


plt.Scatter(X_data, Y_data, c="red", marker="*")       # will give you '*' instead of dots for your points

plt.Scatter(X_data, Y_data, c="red", marker="*", s=150) # 's' to change the size of the points 

""" 
We can also create transparency. 

A case in which we would want to use this is if we have many more data points 
and want to see which datapoints are most repeated (overlaps)

Example below
"""

X_data = np.random.random(1000) * 100
Y_data = np.random.random(1000) * 100

plt.Scatter(X_data, Y_data, c="red", marker="*", s=150, alpha=0.3)

""" 
In the above 0.3 is the amount of transparency. When show you should be able to see
if the points overlap, darker areas, lighter areas if no overlap.
"""

# Line Chart

""" 
Sometimes we don't want to plot datapoints but we want a line (e.g. time series data)

Below is an example where I want to visualise the weight change of a person over the years
"""

# Uisng an oridinary python list  

years = [2006 +x for x in range(16)]

weight = [80, 83, 84, 85, 86, 82, 81, 79, 83, 80,
82, 82, 83, 81, 80, 79]

plt.plot(years, weight)     # default plot of plt function is a line so we don't need plt.line

#plt.show()

plt.plot(years, weight, c="r") # r is unique so will change line to red

plt.plot(years, weight, c="r", lw=3) # lw is for line width (higher number - thicker)

plt.plot(years, weight, c="r", lw=1, linestyle="--") # changing line type - dash etc.

""" 
You can also do the above with a positional parameter.

Shown below
"""

plt.plot(years, weight, "r--", lw=2)  # should have red-dashed line

# Bar Chart

""" 
Now sometimes we want to look at categorical data. For example we might have
a poll and as a question, and you want to visualise the responses.

"""

X = ["C++", "C#", "Python", "Java", "Go"]
Y = [20, 50, 140, 1, 45]  # the Y data here represents the votes

plt.Bar(X, Y)
#plt.show()

plt.Bar(X, Y, color="b", align="edge") # align will tell us where we want the bar to start

plt.Bar(X, Y, color="blue", align="edge", width=0.5)  # thinner bars

plt.Bar(X, Y, color="blue", align="edge", width=0.5, edgecolor="green") # colour of edges

# Histograms

""" 
Now bar charts are used to display categorical data. Different categories, who chose them.
Whereas histograms are used to show the distribution of data within a dataset. For example, 
the distribution of salaries between employees.

In the code below, I am generating some data for a normal distribution.
The first number represents the mean, the second number the standard deviation and the third the 
number of different ages generated. 

Standard deviation - a measure of the distribution of the data in relation to the mean.
Low/small standard deviation - indicates data is clustered around the mean whereas higher/bigger indicates
that the data is more spread out.

"""

ages = np.random.normal(20, 1.5, 1000)

plt.hist(ages)
#plt.show()

""" 
We can also customise the bins

Bins in histograms are the intervals that divide the range of values into groups
Each bar in a histogram typically covers a range of numeric values called a bin (or a class)

Shown in below code
"""
 
plt.hist(ages, Bins=20)

plt.hist(ages, Bins=[ages.min(), 18, 21, ages.max()])

""" 
Above basically shows a histogram of ages.

From 0 - ages.min() will be a bin, from ages.min() to 18 will be another bin and etc.
"""

""" 
We can also plot cumulative histograms

These type of histograms show the cumulative proportion below a certain value, whereas
a histogram shows the data points in pre-defined intervals.
"""

plt.hist(ages, bins=20, cumulative=True)

#Pie Chart

""" 
Sometimes we have independent categories. For example you are this or that,
and there are no overlaps.

It then makes sense to show the ratio of different people

For example below we'll use the example: Which programming language is your favourite in which
you can only have one choice
"""

langs = ["Python", "C++", "Java", "C#", "Go"]
votes = [50, 24, 14, 6, 17]

plt.Pie(votes, labels=langs)
#plt.show()

explodes = [0, 0, 0, 0.2, 0]    # exploding out c#, ignoring the rest

plt.Pie(votes, labels=langs, explode=explodes)

""" 
You can also write the percentage, inside or outside of the pie chart.

How to do this is shown below
"""

# format is '%.2f%%' .2f means 2 decimal places
plt.Pie(votes, labels=langs, explode=explodes, autopct="%.2f%%")

# pctdistance to specify the distance of the percentage
plt.Pie(votes, labels=langs, explode=explodes, autopct="%.2f%%", pctdistance=1.8)

# choose a starting angle
plt.Pie(votes, labels=langs, explode=explodes, autopct="%.2f%%", pctdistance=1.8, startangle=90)

# BoxPlots

""" 
Statistical plot showing you the different quartiles - median, maximum, minimum 
and showing you outliers

The first number in the tuple/array below shows the default. The second 
number shows the standard deviation and the third the number of data points. So in this example 172cm as default,
8 as the deviation and 300 heights are generated.
"""

heights = np.random.normal(172, 8, 300)

plt.boxplot(heights)
#plt.show()

# sections for the boxplot
first = np.linespace(0, 10, 25)
second = np.linespace(10, 200, 25)
third = np.linespace(200, 210, 25)
fourth = np.linespace(210, 230, 25)

data = np.concatenate((first, second, third, fourth))

plt.boxplot(data)
#plt.show()

"""
There are many more plot diagrams you can use in matplotlib but I've gone 
over the most essential ones for you to know. There's so many more like heatmaps etc that you can create 
from scratch (I'd reccommend using seaborn for heatmaps instead of making your own but I can't come up with
another example)
"""

# Plot Customisation - axis, titles, etc.

years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
income = [55, 56, 62, 61, 72, 72, 73, 75]

income_ticks = list(range(50, 81, 2))

plt.plot(years, income)
#plt.show()

""" 
Income ticks is for the y- axis so that we have the actual ticks we want to plot. We 
will use them in a second. I have put the range from 50 to 81 because of our range
of values in 'income'.

Below is us making titles for the diagram, labels for the axis and more!
"""

plt.title("Income of John (in BGP)", fontsize=30, fontname="serif")
plt.xlabel("Year")
ply.ylabel("Yearly Income in GBP")

plt.yticks(income_ticks, [f"{x}k GBP" for x in income_ticks])

plt.plot(years, income)

""" 
You can also add a 'legend' to the plot.
You might have different things you want to plot

Call plot function multiple times - seeing different line charts (in this specific example)
in the same plot. 

To explain the difference between them you can use a legend.

Below is an example!
"""

stock_a = [100, 102, 99, 101, 101, 100, 102]
stock_b = [90, 95, 102, 104, 105, 103, 109]
stock_c = [116, 115, 100, 105, 100, 98, 95]

plt.plot(stock_a, label="Company1")
plt.plot(stock_b, label="Company2")
plt.plot(stock_c, label="Company3")

plt.legend()

""" 
Important to note - If I only provide the x values (so like above plt.plot(stock_a)),
the x values will be automatically generate with numbers (1, 2, 3, etc.)

Adding labels to each line so you can label what they're for.
Without the 'plt.legend()' line nothing will change but when you add that line,
in the top right corner by default you'll get a legend. 

Below the 'loc' keyword will just change th position of the legend.
I'm setting it to lower right
"""

plt.legend(loc="lower right")

""" 
This works for all the different plot types aswell. 

Example of pie chart below
"""

votes = [10, 2, 5, 16, 22]
people = ["A", "B", "C", "D", "E"]

plt.pie(votes, labels=None)
plt.legend(labels=people)

# Style sheets
from matplotlib import Style

style.use("ggplot") # gets a particular style sheet

style.use("dark_background")    # dark theme

"""
Full list of style sheets available online - also can create your own
"""

# Multiple figures

""" 
Sometimes I want to plot multiple diagrams side by side.

Code below will give me two independent windows of id=ndepend data visauls
This is ONE way of doing it
"""

x1, y1 = np.random(100), np.random(100)
x2, y2 = np.arange(100), np.random.random(100)

plt.figure(1)
plt.scatter(x1, y1)

plt.figure(2)
plt.plot(x2, y2)

#plt.show()

# SubPlots

""" 
One window/one figure but you have different subplots in that one window
"""

x = np.arange(100)

fig, axs = plt.subplots(2, 2)   # 2x2 grid

"""
This gives us one figure with 4 different axis (0,0), (0,1), (1,0), (1,1)
A grid of 4 subplots in one figure 

We can access the individual points by...(see below)
"""

axs[0, 0].plot(x, np.sin(x))
axs[0, 0].set_title("Sine wave")    # we have to say set_title

axs[0, 1].plot(x, np.cos(x))        # row 0, column 1
axs[0, 1].set_title("Cosine wave")

axs[1, 0].plot(x, np.random.random(100))
axs[1, 0].set_title("Random function")

axs[1, 1].plot(x, np.log(x))
axs[1, 1].set_title("Log function") 
axs[1, 1].set_xlabel("Test")

""" 
The entire figure can have a super title aswell

Shown below
"""

fig.suptitle("Four Plots")

# Exporting plots 

""" 
How to export a graphic instead of show...(Shown below)

We can also alter the quality of the picture using dpi

We also can set transparency. 'True' - without the white background

We also have 'bbox' (boundary box) - tight_inches cuts away the boundary padding - more space
towards the edges

Tight layout structures everything so there's no overlap
"""
plt.tight_layout()  # stops the overlap between the ticks and titles

plt.savefig("fourplots.png", dpi=300, transparent=False, bbox_inches ="tight")

# There's many more other things you can do, you can look online! (Documentation)

# 3D PLOTTING

""" 
First we're going to create a new axis
"""

ax = plt.axes(projection="3d")

""" 
Now we can define x, y and z values
"""

x = np.random.random(100)
Y = np.random.random(100)
Z = np.random.random(100)

""" 
And now instead of scattering in 2 dimentsions I can now scatter im 3
"""

ax.scatter(x, y, z)
ax.set_title("3D Plot")
ax.set_xlabel("Test")

#plt.show()

""" 
Everything we have covered with the labelling and styling etc also 
work in 3d so I won't be showing those again...

Now we can also plot a line plot instead of scatter. It makes more sense to use different valuees (not random)
"""

x = np.arange(0, 50, 0.1)
Y = np.sin(x)
Z = np.cos(x)

ax.plot(x, y, z)

""" 
Surface plot - We want to have actual surfaces being plotted

To do this we generate x and y values and then create a 'mesh' grid
so called combination.
How we do that is shown below...
"""
x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)

""" 
In order to actually create the mesh grid we....
"""

X, Y = np.meshgrid(x, y)

Z = np.sin(X) * np.cos(Y)

ax.plot_surface(Z, Y, Z, cmap="spectral")
#plot.show()

""" 
In the bottom right corner, as we rotate our 3D diagram around we should be able to see
2 chnaging values that represent the azimuth (a measurement of direction in degrees) and the elevation

These two values can also be set by you (shown below) and you will be presented with a default 2D diagram of a still image of 
what the 3D diagram looks like at those two specific values - CAN STILL BE MOVED AROUND
"""

ax.view_init(azim=0, elev=90)

# Animating Plots

import random

""" 
We're going to flip a coin (50/50) and keep track of the results
"""

head_tails = [0, 0]

for _ in range(100000):     # 100000 coin flips 
    head_tails[random.randint(0, 1)] += 1 # headtails either x,y incrreased by 1 at every iteration
    plt.bar(["Heads", "Tails"], head_tails, color=["red", "blue"])
    plt.pause(0.001)    # like show, but animates - continues with plotting
#plt.show() 

""" 
Due to the large amound of numbers after a while the animation should eventually show 
equal bars, as it's a 50/50 precedure

using 'pause' is what animates the function
"""

