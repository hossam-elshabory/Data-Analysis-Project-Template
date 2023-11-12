########################### PLOTTING UTILITY CLASS (OBJECT) ###########################


class Plot:
    """
    A class for creating plots using Seaborn with the strategy design pattern.

    Attributes:
    -----------
    data : Union[pd.DataFrame, pd.Series]
        The input data for plotting.
    """

    def __init__(self, data: Union[pd.DataFrame, pd.Series]):
        """
        Parameters:
        -----------
        data : Union[pd.DataFrame, pd.Series]
            The input data for plotting.
        """
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise ValueError("Invalid input type. Data must be a DataFrame or Series.")
        self.data = data

    def apply_plot(self, strategy: Callable, **kwargs) -> "Plot":
        """
        Apply a plot strategy to the data.

        Parameters:
        -----------
        strategy : Callable
            The plot strategy function.
        **kwargs :
            Additional keyword arguments to be passed to the strategy function.

        Returns:
        --------
        Plot
            Returns self for method chaining.
        """
        strategy(self.data, **kwargs)
        return self

    def show_plot(self) -> None:
        """
        Display the plot.

        Returns:
        --------
        None
        """
        plt.show()


########################### PLOTTING FUNCTIONS (STRATEGIES) ###########################


def barplot(
    data: Union[pd.DataFrame, pd.Series],
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: Union[List[int], Tuple[int, int]] = (10, 5),
    palette: str = "Blues_r",
    text_offset: int = 130,
    title_fontsize: int = 15,
    label_fontsize: int = 11,
    hide_axis: str = "none",
    label_value: str = "label",
) -> None:
    """
    Create a horizontal bar plot strategy with customizable parameters.

    Parameters:
    -----------
    data : Union[pd.DataFrame, pd.Series]
        The input data for plotting.
    title : str
        The title of the plot.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    figsize : Union[List[int], Tuple[int, int]], optional
        Size of the figure (default is (10, 5)).
    palette : str, optional
        Seaborn color palette for the plot (default is "Blues_r").
    text_offset : int, optional
        Offset for text labels showing the popularity of each genre (default is 130).
    title_fontsize : int, optional
        Font size for the plot title (default is 15).
    label_fontsize : int, optional
        Font size for the y-axis tick labels (default is 11).
    hide_axis : str, optional
        Specify which axis to hide. Options: 'none', 'x', 'y' (default is 'none').
    label_value : str, optional
        String value for the text next to the bars in the chart.

    Returns:
    --------
    None
    """
    # Setting the plot's figure size
    fig, ax = plt.subplots(figsize=figsize)

    # Plotting the horizontal bar chart
    sns.barplot(x=data[:10].values, y=data[:10].index, palette=palette, ax=ax)

    # Adding labels next bars
    if label_value is not None:
        for i, v in enumerate(data[:10].values):
            ax.text(
                v + text_offset / 2,
                i,
                str(label_value),
                color="black",
                ha="center",
                va="center",
            )

    # Hide the specified axis
    if hide_axis == "x":
        ax.set_xticks([])
        ax.set_xticklabels([])
    elif hide_axis == "y":
        ax.set_yticks([])
        ax.set_yticklabels([])

    # Setting the y-axis ticks and labels
    if hide_axis != "y":
        ax.tick_params(axis="y", labelsize=label_fontsize)

    # Setting the title and axis labels
    ax.set_title(title, fontsize=title_fontsize, pad=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Removing the spines from the right and top sides of the plot
    sns.despine(right=True, top=True, bottom=True)

    # Showing the plot
    plt.show()


def scatterplot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    figsize: Tuple[int, int] = (10, 5),
    s: int = 50,
) -> None:
    """
    Create a scatterplot strategy for visualizing the relationship between two columns in a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame for which the scatterplot will be created.
    x_col : str
        The column for the x-axis.
    y_col : str
        The column for the y-axis.
    title : str, optional
        The title of the plot.
    figsize : Tuple[int, int], optional
        Size of the figure (default is (10, 5)).
    s : int, optional
        Marker size in the scatterplot (default is 50).

    Returns:
    --------
    None
    """
    # Setting the plot's figure size
    fig, ax = plt.subplots(figsize=figsize)

    # Plotting the scatterplot
    ax = sns.scatterplot(data=df, x=x_col, y=y_col, s=s)

    # Despine the plot from the right and top
    sns.despine(ax=ax, top=True, right=True)

    # Add a descriptive title and increase its size and add padding
    ax.set_title(title, fontsize=20, pad=25)

    # Increase the size of the x-axis and y-axis ticks
    ax.tick_params(axis="both", labelsize=12)

    # Increase the size of the x-axis and y-axis labels and add paddings
    ax.set_xlabel(x_col, fontsize=15, labelpad=20)
    ax.set_ylabel(y_col, fontsize=15, labelpad=20)

    # Show the plot
    plt.show()


def heatmap(
    df: pd.DataFrame,
    title: str = "Correlation Heatmap",
    figsize: Tuple[int, int] = (15, 5),
    cmap: str = "Blues",
    annot: bool = True,
    annot_kws: dict = {"size": 12},
) -> None:
    """
    Create a heatmap strategy for visualizing the correlation matrix of a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame for which the correlation matrix will be visualized.
    title : str, optional
        The title of the plot (default is "Correlation Heatmap").
    figsize : Tuple[int, int], optional
        Size of the figure (default is (15, 5)).
    cmap : str, optional
        Seaborn color palette for the heatmap (default is "Blues").
    annot : bool, optional
        Whether to annotate the heatmap with correlation values (default is True).
    annot_kws : dict, optional
        Additional keyword arguments for annotating the heatmap (default is {"size": 12}).

    Returns:
    --------
    None
    """
    # Calculating the correlation matrix
    corr_matrix = df.corr()

    # Setting up the figure size
    fig, ax = plt.subplots(figsize=figsize)

    # Creating the heatmap
    sns.heatmap(corr_matrix, annot=annot, cmap=cmap, annot_kws=annot_kws, ax=ax)

    # Setting up the title
    ax.set_title(title, fontsize=20, pad=25)

    # Setting the x-axis and y-axis ticks sizes
    ax.tick_params(axis="both", labelsize=12)

    # Showing the plot
    plt.show()
