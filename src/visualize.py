"""Visualisation codes for basic dataset description and for HMM and CPD
Author: Maris Sala
Date: 14th Dec 2021
"""
import pandas as pd
import numpy as np
from icecream import ic

from scipy.signal import savgol_filter

import networkx as nx
import seaborn as sns; sns.set()
import pyplot_themes as themes
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.cbook.deprecation import MatplotlibDeprecationWarning

##################################################################################
### PLOT SETTINGS                                                              ###
##################################################################################
class PlotSettings:
    @staticmethod
    def basePlotSettings(fontsize: int, 
                            if_palette: bool):
        """Controls the base settings for all seaborn plots
        Args:
        fontsize: font size for ticks
        if_palette: if True then returns a colorblind friendly color palette,
                    else returns nothing
        
        Returns:
        fig, ax1 and palette
        """
        matplotlib.rc('ytick', labelsize=fontsize)
        matplotlib.rc('xtick', labelsize=fontsize)
        themes.theme_minimal(grid=False, ticks=False, fontsize=fontsize)
        a4_dims = (25,15)
        
        if if_palette:
            #          0 black      1 orange  2 L blue   3 green    4 L orange  5 D blue  6 D orange 7 purple
            palette = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
        else:
            palette = 0
        
        fig, (ax1) = plt.subplots(1,1, figsize=a4_dims)
        sns.set(font_scale = 2)

        return fig, ax1, palette

    @staticmethod
    def latePlotSettings(fig, 
                            ax1, 
                            if_dates:bool):
        """Configures plot settings after plotting. Removes x and y labels,
        sets ylim, fontsizes for labels
        Args:
        fig: matplotlib seaborn figure
        ax1: matplotlib seaborn axis
        if_dates: whether x axis is a time series

        Returns:
        Updated fig and ax1
        """
        ax1.set(xlabel="", ylabel = "")
        ax1.xaxis.get_label().set_fontsize(40)
        ax1.yaxis.get_label().set_fontsize(40)
        ax1.tick_params(labelsize=30)

        ax1.grid(color='white', linestyle='-', linewidth=0.5, which= "both")
        if if_dates:
            # Define the date format
            ax1.xaxis_date(tz="UTC")
            date_form = mdates.DateFormatter("%b-%Y")#("%d-%b")
            ax1.xaxis.set_major_formatter(date_form)

        ax1.set(ylim=(0, None))

        for tick_label in ax1.axes.get_yticklabels():
            tick_label.set_color("black")
        for tick_label in ax1.axes.get_xticklabels():
            tick_label.set_color("black")

        ax1.tick_params(labelsize=30)

        return fig, ax1

    @staticmethod
    def lateBarPlotSettings(fig, 
                                ax1):
        """Sets settings for barplots after plotting. 
        Removes labels on x and y axis
        Args:
        fig: matplotlib seaborn figure
        ax1: matplotlib seaborn axis

        Returns:
        Updated fig and ax1
        """
        ax1.set(xlabel="", ylabel = "")
        ax1.xaxis.get_label().set_fontsize(20)
        ax1.yaxis.get_label().set_fontsize(40)
        ax1.tick_params(labelsize=30)
        for tick_label in ax1.axes.get_yticklabels():
            tick_label.set_color("black")
        for tick_label in ax1.axes.get_xticklabels():
            tick_label.set_color("black")
        return fig, ax1
    
    @staticmethod
    def addLegend(fig, ax1):
        """Adds legend to figures, prefer not to use
        """
        leg = plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', facecolor='white')
        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            legobj.set_linewidth(10.0)
        return fig, ax1
    
    @staticmethod
    def jitter(values,j):
        """Creates slightly altered numbers to jitter them in a scatterplot
        Args:
        values: list of values to jitter (column from DataFrame on y-axis)
        j: dimensions

        Returns:
        jittered values
        """
        return values + np.random.normal(j,0.1,values.shape)

##################################################################################
### VISUALS FUNCTIONS FOR DESCRIBING DF                                        ###
##################################################################################

class PlotVisuals:
    @staticmethod
    def basicsLineplot(df, 
                        root_path:str, 
                        datatype:str,
                        comment:str):
        """Generates a line plot with numeric statistics of the dataset
        over time
        Args:
        df: pandas DataFrame, needs column "date" in Datetime format
        root_path: path of the current directory
        datatype: "posts" or "comments" for the Facebook data
        comment: anything noteworthy to add to filename

        Returns:
        Saves the figure in the local directory
        """
        ic("[PLOT] Basic lineplot started")
        df = df.groupby("date").agg({"id": 'sum', 
                                    'comment_count': 'sum',
                                    'likes_count': 'sum',
                                    'shares_count': 'sum'}).reset_index()

        ic(df.head())
        
        fig, ax1, palette = PlotSettings.basePlotSettings(fontsize=30, if_palette = True)
        linewidth=3

        ax1 = sns.lineplot(x="date", y="comment_count",
                        color = palette[1], 
                        #label = "Comments",
                            linewidth=linewidth, data = df)

        ax1 = sns.lineplot(x="date", y="likes_count",
                        color = palette[2], 
                        #label = "Likes",
                            linewidth=linewidth, data = df)
        
        ax1 = sns.lineplot(x="date", y="shares_count",
                        color = palette[3], 
                        #label = "Shares",
                            linewidth=linewidth, data = df)
        
        ax1 = sns.lineplot(x="date", y="id",
                        color = palette[0], 
                        #label = "Number of posts",
                        #s = 200,
                        linewidth=linewidth, 
                        data = df)
        
        fig, ax1 = PlotSettings.latePlotSettings(fig, ax1, if_dates = True)

        #ic("Add legend")
        #fig, ax1 = PlotSettings.addLegend(fig, ax1)
        
        plot_name = f"{root_path}out/fig/specs/{datatype}_{comment}_word_count.png"
        #fig.savefig(plot_name, bbox_extra_artists=(leg,), bbox_inches='tight') # in case a legend
        fig.savefig(plot_name, bbox_inches='tight')
        ic("[PLOT] Basic lineplot finished")

    @staticmethod
    def posts_per_day_per_group_scatterplot(ori_df, 
                                            root_path:str, 
                                            datatype:str,
                                            comment:str):
        """Generates a scatter plot with posts per day
        Args:
        ori_df: pandas DataFrame, needs column "date" in Datetime format
        root_path: path of the current directory
        datatype: "posts" or "comments" for the Facebook data

        Returns:
        Saves the figure in the local directory
        """
        ic("posts_per_day_per_group_scatterplot")
        df = ori_df.groupby(["date", "group_id"]).agg({"id": 'sum'}).reset_index()
        ic(df.head())
        ic(df.id.unique())
        ic(df.describe())

        ic("Base plot settings")
        fig, ax1, palette = PlotSettings.basePlotSettings(fontsize=30, if_palette = True)
        ic("Scatter plot")

        ax1 = sns.scatterplot(x="date", y="id", #y=jitter(df["id"],2), 
                            hue="group_id", 
                            s = 50,
                            legend=False, data = df)
        
        ax1 = sns.lineplot(x="date", y="id", #y=jitter(df["id"],2), 
                            hue="group_id", 
                            linewidth = 3,
                            legend=False, data = df)
        
        ic("Late plot settings")
        fig, ax1 = PlotSettings.latePlotSettings(fig, ax1, if_dates = True)

        ic("Save image")
        plot_name = f"{root_path}out/fig/specs/{datatype}_{comment}_posts_per_day_per_group_scatter.png"
        fig.savefig(plot_name, bbox_inches='tight')
        
        ic("Save figure done\n------------------\n")

    @staticmethod
    def posts_per_group_barplot(ori_df, 
                                root_path:str, 
                                datatype:str,
                                comment:str):
        """Generates a bar plot with posts per group
        Args:
        ori_df: pandas DataFrame
        root_path: path of the current directory
        datatype: "posts" or "comments" for the Facebook data

        Returns:
        Saves the figure in the local directory
        """
        df = ori_df.groupby("group_id").agg({"id": 'sum'}).reset_index()
        n = 50
        ic(df.describe()) # Report this in the paper!
        df = df.sort_values('id', ascending=False)[0:n]

        ic("Base plot settings")
        fig, ax1, palette = PlotSettings.basePlotSettings(fontsize=30, if_palette = True)
        ic("Bar plot")
        df["group_id"] = df["group_id"].astype(int).astype(str)
        ax1 = sns.barplot(y="group_id", x="id",
                        color = palette[7], 
                        #order = df.sort_values('id', ascending=False).group_id,
                        data = df)

        ic("Late plot settings")
        fig, ax1 = PlotSettings.lateBarPlotSettings(fig, ax1)

        ax1.tick_params(labelsize=15)

        ic("Save image")
        plot_name = f"{root_path}out/fig/specs/{datatype}_posts_per_{n}_groups.png"
        fig.savefig(plot_name, bbox_inches='tight')
        ic("Save figure done\n------------------\n")

        df = ori_df.groupby("group_id").agg({"id": 'sum'}).reset_index()
        ic(df.describe()) # Report this in the paper!
        df = df.sort_values('id', ascending=False)

        ic("Base plot settings")
        fig, ax1, palette = PlotSettings.basePlotSettings(fontsize=30, if_palette = True)
        ic("Bar plot")
        df["group_id"] = df["group_id"].astype(int).astype(str)
        ax1 = sns.barplot(x="group_id", y="id",
                        color = palette[7], 
                        #order = df.sort_values('id', ascending=False).group_id,
                        data = df)

        ic("Late plot settings")
        fig, ax1 = PlotSettings.lateBarPlotSettings(fig, ax1)

        #ax1.tick_params(labelsize=15)
        #ax1.set_xticklabels(ax1.get_xticklabels(),rotation = 50)
        ax1.set(xticklabels=[])  
        ax1.set(xlabel=None)
        ax1.tick_params(bottom=False)

        ic("Save image")
        plot_name = f"{root_path}out/fig/specs/{datatype}_{comment}_posts_per_group.png"
        fig.savefig(plot_name, bbox_inches='tight')
        ic("Save figure done\n------------------\n")

    @staticmethod
    def unique_users_per_group_barplot(ori_df, 
                                    root_path:str, 
                                    datatype:str,
                                    comment:str):
        """Generates a bar plot with unique users per group
        Args:
        ori_df: pandas DataFrame
        root_path: path of the current directory
        datatype: "posts" or "comments" for the Facebook data

        Returns:
        Saves the figure in the local directory
        """
        df = ori_df.groupby("group_id").agg({"user_id": 'sum'}).reset_index()
        ic(df.describe()) #Report this in the paper!
        n=50
        df = df.sort_values("user_id", ascending=False)[:n]

        ic("Base plot settings")
        fig, ax1, palette = PlotSettings.basePlotSettings(fontsize=30, if_palette = True)
        ic("Bar plot")
        df["group_id"] = df["group_id"].astype(int).astype(str)
        ax1 = sns.barplot(y="group_id", x="user_id",
                        color = palette[7], 
                        #order = df.sort_values('user_id', ascending=False).group_id,
                        data = df)

        ic("Late plot settings")
        fig, ax1 = PlotSettings.lateBarPlotSettings(fig, ax1)

        ax1.tick_params(labelsize=15)

        ic("Save image")
        plot_name = f"{root_path}out/fig/specs/{datatype}_{comment}_unique_users_per_{n}_groups.png"
        fig.savefig(plot_name, bbox_inches='tight')
        ic("Save figure done\n------------------\n")

        df = ori_df.groupby("group_id").agg({"user_id": 'sum'}).reset_index()
        ic(df.describe()) #Report this in the paper!
        df = df.sort_values("user_id", ascending=False)

        ic("Base plot settings")
        fig, ax1, palette = PlotSettings.basePlotSettings(fontsize=30, if_palette = True)
        ic("Bar plot")
        df["group_id"] = df["group_id"].astype(int).astype(str)
        ax1 = sns.barplot(x="group_id", y="user_id",
                        color = palette[7], 
                        #order = df.sort_values('user_id', ascending=False).group_id,
                        data = df)

        ic("Late plot settings")
        fig, ax1 = PlotSettings.lateBarPlotSettings(fig, ax1)

        #ax1.tick_params(labelsize=15)
        #ax1.set_xticklabels(ax1.get_xticklabels(),rotation = 50)
        ax1.set(xticklabels=[])  
        ax1.set(xlabel=None)
        ax1.tick_params(bottom=False)

        ic("Save image")
        plot_name = f"{root_path}out/fig/specs/{datatype}_{comment}_unique_users_per_group.png"
        fig.savefig(plot_name, bbox_inches='tight')
        ic("Save figure done\n------------------\n")

    @staticmethod
    def posts_users_scatterplot(ori_df, 
                                root_path:str, 
                                datatype:str,
                                comment:str):
        """Generates a scatter plot with posts vs users
        Args:
        ori_df: pandas DataFrame
        root_path: path of the current directory
        datatype: "posts" or "comments" for the Facebook data

        Returns:
        Saves the figure in the local directory
        """
        ic(len(ori_df.group_id.unique()))
        users = ori_df.groupby("group_id").agg({"user_id": 'sum'}).reset_index()
        ic(len(users.group_id.unique()))
        posts = ori_df.groupby("group_id").agg({"id": 'sum'}).reset_index()
        ic(len(posts.group_id.unique()))
        
        df = pd.merge(left=users, right=posts, on="group_id")
        ic(df.head())
        ic(len(df.group_id.unique()))

        ic("Base plot settings")
        fig, ax1, palette = PlotSettings.basePlotSettings(fontsize=30, if_palette = True)
        ic("Scatter plot")
        
        ax1 = sns.scatterplot(x="id", y="user_id",
                        color = palette[0],
                        s = 150, 
                        #order = df.sort_values('user_id', ascending=False).group_id,
                        data = df)
        ax1 = sns.regplot(x="id", y="user_id", data=df)

        ic("Late plot settings")
        fig, ax1 = PlotSettings.latePlotSettings(fig, ax1, if_dates = False)

        ax1.set_xlabel("Unique posts per group", fontsize = 40, color="black")
        ax1.set_ylabel("Unique users per group", fontsize = 40, color="black")

        ic("Save image")
        plot_name = f"{root_path}out/fig/specs/{datatype}_{comment}_posts_vs_unique_users.png"
        fig.savefig(plot_name, bbox_inches='tight')
        ic("Save figure done\n------------------\n")

    @staticmethod
    def unique_users_over_time_lineplot(ori_df, 
                                        root_path:str, 
                                        datatype:str,
                                        comment:str):
        """Generates a line plot with unique users over time
        Args:
        ori_df: pandas DataFrame, needs column "date" in Datetime format
        root_path: path of the current directory
        datatype: "posts" or "comments" for the Facebook data

        Returns:
        Saves the figure in the local directory
        """
        df = ori_df.groupby("date").agg({"user_id": 'sum'}).reset_index()
        ic(df.describe())

        ic("Base plot settings")
        fig, ax1, palette = PlotSettings.basePlotSettings(fontsize=30, if_palette = True)
        ic("Line plot")
        ax1 = sns.lineplot(x="date", y="user_id",
                        palette = palette[0], 
                            linewidth = 3, data = df)
        
        ic("Late plot settings")
        fig, ax1 = PlotSettings.latePlotSettings(fig, ax1, if_dates = True)
        # Add a line to signify where there are less than x users
        ax1.axhline(100, color="red", linewidth=2)

        ic("Save image")
        plot_name = f"{root_path}out/fig/specs/{datatype}_{comment}_unique_users_over_time.png"
        fig.savefig(plot_name, bbox_inches='tight')
        
        ic("Save figure done\n------------------\n")

    @staticmethod
    def total_lifespan_per_group_pointplot(ori_df, 
                                            root_path:str, 
                                            datatype:str,
                                            comment:str):
        """Generates a point plot with starting and ending dates for group activity
        Args:
        ori_df: pandas DataFrame, needs column "date" in Datetime format
        root_path: path of the current directory
        datatype: "posts" or "comments" for the Facebook data

        Returns:
        Saves the figure in the local directory
        """
        df = ori_df.groupby("group_id").agg({"date": ['min', 'max']}).reset_index()
        
        frame = {"group_id": df["group_id"], "date": df["date", "min"]}
        mindates = pd.DataFrame(data=frame)
        frame = {"group_id": df["group_id"], "date": df["date", "max"]}
        maxdates = pd.DataFrame(data=frame)
        
        combined = pd.concat([mindates, maxdates])
        combined["date"] = pd.to_datetime(combined["date"])
        combined['days'] = (maxdates['date'] - mindates['date']).dt.days
        combined = combined.sort_values("date").reset_index(drop=True).reset_index()

        ic("Base plot settings")
        fig, ax1, palette = PlotSettings.basePlotSettings(fontsize=30, if_palette = True)
        ic("Line & Scatter plot")
        ic(len(combined["group_id"]))
        ax1 = sns.lineplot(x="date", y="days", hue="group_id",
                            linewidth = 3,
                            data = combined, legend = False)
        ax1 = sns.scatterplot(x="date", y="days", hue="group_id",
                            s = 150,
                            data = combined, legend = False)
        
        ic("Late plot settings")
        fig, ax1 = PlotSettings.latePlotSettings(fig, ax1, if_dates = True)

        filename = f"{root_path}res/min_max_dates_per_group.csv"
        combined.to_csv(filename, index=False)

        ic("Save image")
        plot_name = f"{root_path}out/fig/specs/{datatype}_{comment}_total_lifespan_per_group.png"
        fig.savefig(plot_name, bbox_inches='tight')
        
        ic("Save figure done\n------------------\n")

    @staticmethod
    def visualize_posts_per_week(root_path, 
                                datatype,
                                comment:str,
                                df=False):
        if not df:
            filename = f"{root_path}res/weekly_downsampled.csv"
            df = pd.read_csv(filename, sep=";")
        
        df["date"] = pd.to_datetime(df["date"])
        ic("[INFO] Visualize")
        ic("Base plot settings")
        fig, ax1, palette = PlotSettings.basePlotSettings(fontsize=30, if_palette = True)
        ic("Line plot")
        
        ax1 = sns.scatterplot(x="date", y="id", hue="group_id",
                        #palette = palette[4], 
                            s = 50, data = df, legend=False)
        ax1 = sns.lineplot(x="date", y="id", hue="group_id",
                        #palette = palette[4], 
                            linewidth = 1, data = df, legend=False)
        
        ic("Late plot settings")
        fig, ax1 = PlotSettings.latePlotSettings(fig, ax1, if_dates = True)
        
        ax1.xaxis_date(tz="UTC")
        #date_form = mdates.DateFormatter("%b-%Y")
        #ax1.xaxis.set_major_formatter(date_form)

        ic("Save image")
        plot_name = f"{root_path}out/fig/specs/{datatype}_{comment}_posts_per_week_per_group_scatterlinepot.png"
        fig.savefig(plot_name, bbox_inches='tight')
        
        ic("DONE")

##################################################################################
### VISUALS FUNCTIONS FOR HMMs and CPD                                         ###
##################################################################################

class PlotHMMCPD:
    @staticmethod
    def jitter(values,j):
        """Creates slightly altered numbers to jitter them in a scatterplot
        Args:
        values: list of values to jitter (column from DataFrame on y-axis)
        j: dimensions

        Returns:
        jittered values
        """
        return values + np.random.normal(j,0.1,len(values))

    @staticmethod
    def incr(lst:list, i:int, type:str):
        """List of two elements, increase one number
        """
        el1 = lst[0]
        el2 = lst[1]
        if type == "1st":
            return [el1+i, el2] #[x+i for x in lst]
        elif type == "2nd":
            return [el1, el2+i]
        elif type == "both":
            el1 = PlotHMMCPD.jitter([el1],i)[0]
            el2 = PlotHMMCPD.jitter([el1],i)[0]
            return [el1, el2]

    @staticmethod
    def visualize(X, Z):
        ic(len(X))
        x = np.linspace(0, len(X), num=len(X))
        ic(len(x))
        y1 = X
        #y2 = Z

        plt.figure(num = 3, figsize=(8, 5))
        # larger window length means smaller signal
        plt.plot(x, savgol_filter(y1, 101, 3), alpha = 0.3)
        plt.plot(x, savgol_filter(y1, 301, 1), 
                color='red',   
                linewidth=1
                )

        plt.savefig("fig1.png")

    @staticmethod
    def visualize_HMM(OUT_PATH, comment, group_id, novelty, resonance, nov_states, res_states):
        palette = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

        #### NOVELTY ####
        df = pd.DataFrame(dict(novelty=novelty, state=nov_states)).reset_index()
        df.columns = ["time", "novelty", "state"]
        fig, ax = plt.subplots()
        my_states = list(set(nov_states))
        colors = {}
        for i in my_states:
            colors[i] = palette[i]
        ax.scatter(df['time'], df['novelty'], 
                    c=df['state'].map(colors),
                    s=0.2)
        ax.set(ylim=(0,1))
        filename = f"{OUT_PATH}out/fig/hmm/{group_id}_{comment}_HMM_gaussian_novelty.png"
        plt.savefig(filename)
        ic("[INFO] Novelty figure done")

        #### RESONANCE ####
        df = pd.DataFrame(dict(resonance=resonance, state=res_states)).reset_index()
        df.columns = ["time", "resonance", "state"]
        fig, ax = plt.subplots()
        my_states = list(set(res_states))
        colors = {}
        for i in my_states:
            colors[i] = palette[i]
        ax.scatter(df['time'], df['resonance'], 
                    c=df['state'].map(colors),
                    s=0.2)
        ax.set(ylim=(-1,1))
        filename = f"{OUT_PATH}out/fig/hmm/{group_id}_{comment}_HMM_gaussian_resonance.png"
        plt.savefig(filename)
        ic("[INFO] Resonance figure done")

    @staticmethod
    def visualize_HMM_CPD(OUT_PATH, comment, group_id, observ_name, observations, states, nr_of_states, change_points):
        palette = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

        df = pd.DataFrame(dict(observation=observations, state=states)).reset_index()
        df.columns = ["time", "observation", "state"]
        fig, ax = plt.subplots()
        my_states = list(set(states))
        colors = {}
        for i in my_states:
            colors[i] = palette[i]
        ax.scatter(df['time'], df['observation'], 
                    c=df['state'].map(colors),
                    s=2)

        ax = sns.lineplot(df['time'], df['observation'], 
                    #c=df['state'].map(colors),
                    linewidth=0.5, alpha=0.3)

        # Add change points
        dates_df = df[df["time"].isin(change_points)].reset_index(drop=True)
        x_coordinates = dates_df['time']
        y_coordinates = dates_df['observation']
        plt.scatter(x_coordinates, y_coordinates, color="none", edgecolor="red",
                s = 25, linewidths = 1)
        #ax.set(ylim=(0,1))
        group_id = str(int(group_id))
        filename = f"{OUT_PATH}out/fig/hmm/{group_id}_{comment}_HMM_{nr_of_states}_states_{observ_name}.png"
        plt.savefig(filename)
        ic("[INFO] HMM CPD figure done", observ_name)
        
        ic(df)
        ic(dates_df)

    @staticmethod
    def visualize_HMM_model(observ_name, transition_matrix, states, nr_of_states, OUT_PATH, group_id, comment):
        """Based on this beautiful guide with fixes: https://vknight.org/unpeudemath/code/2015/11/15/Visualising-markov-chains.html
        """
        palette = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

        G = nx.MultiDiGraph()
        labels={}
        edge_labels={}
        i_list = []
        j_list = []

        #transition_matrix = transition_matrix * nr_of_states
        #ic(transition_matrix)

        for i, origin_state in enumerate(states):
            i = origin_state[0]
            for j, destination_state in enumerate(states):
                j = destination_state[1]

                rate = transition_matrix[i][j]
                if rate > 0.01:
                    G.add_edge(origin_state,
                            destination_state,
                            weight=rate,
                            label="{:.02f}".format(rate))
                    edge_labels[(origin_state, destination_state)] = label="{:.02f}".format(rate)

        plt.figure(figsize=(14,7))
        node_size = 400
        pos = {state:list(state) for state in states}
        nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
        pos2 = {state: PlotHMMCPD.incr(list(state),0.05,"2nd") for state in states}
        nx.draw_networkx_labels(G,pos,font_weight=5, font_color="blue")
        pos2 = {state: list(PlotHMMCPD.jitter(list(state),0.00005)) for state in states}
        nx.draw_networkx_edge_labels(G, pos2, edge_labels)
        plt.axis('off')

        group_id = str(int(group_id))
        filename = f"{OUT_PATH}out/fig/hmm_model/{group_id}_{comment}_HMM_model_{nr_of_states}_states_{observ_name}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
