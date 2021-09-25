---
layout: post
description: An explanation of my development process for a census data shiny app
categories: [markdown]
title: R Shiny Census App
---
# R Shiny Census App

In this blog post, I'll walk through my development process for a [U.S. Census data visualization web app](http://vbakshi.shinyapps.io/census-app) I created using the Shiny package in R.

## Background

I started this project by reading the handbook <a href="https://www.census.gov/content/dam/Census/library/publications/2020/acs/acs_state_local_handbook_2020.pdf">Understanding and Using American Community Survey Data: What State and Local Government Users Need to Know</a> published by the U.S. Census Bureau. I recreated the handbook's first case study in R, in which they make comparisons across geographic areas, create custom geographic areas from census tracts and calculate margins of error for derived estimates for Minnesota Census Tract 5-year earnings estimates. 

During the process of recreating the derived median earnings estimate calculations, I was unable to recreate a key value from the handbook (the Standard Error for the 50% proportion, calculated to be 0.599) because I was unable to deduce the values used in the following formula referenced from page 17 of the <a href="https://www2.census.gov/programs-surveys/acs/tech_docs/pums/accuracy/2015_2019AccuracyPUMS.pdf">PUMS Accuracy of the Data documentation</a>:

![Standard Error equals Design Factor times square root of the product of 95 over 5B and 50 squared]({{ site.baseurl }}/images/se_formula_original.png)

The documentation defines B as the _base_, which is the _calculated weighted total_. I chose the value of 1.3 for the design factor DF since it corresponds to STATE = Minnesota, CHARTYP = Population, CHARACTERISTIC = Person Earnings/Income in the <a href="https://www2.census.gov/programs-surveys/acs/tech_docs/pums/accuracy/2019_PUMS_5yr_Design_Factors.csv">Design Factors CSV published by the Census Bureau</a>.

I called the <a href="https://www.census.gov/programs-surveys/acs/contact.html">Census Bureau Customer Help Center</a> for assistance and was transferred to a member of the ACS Data User Support team with whom I discussed my woes. He was unable to confirm the values of the design factor DF or B, and was unable to pull up the contact information for the statistical methodology team, so I emailed him my questions. After a few email exchanges, the statistical methodology team provided the following:

- DF = 1.3
- B = the total population estimate for which the median is being calculated, which is 82488 for the case study calculation (Minnesota Rural Male Full Time Workers)
- The term 95/5 is associated with the finite population correction factor (100 - f) divided by the sample fraction (f), where f = 5% (later on I note in the documentation that this 95/5 term is based on a 68% confidence interval). The data used in the handbook case study is from 5-year estimates. 1-year estimates sample 2.5% of the population, so the 5-year estimates represent a 5 * 2.5 = 12.5% sample. Instead of 95/5, the ratio becomes (100 - 12.5)/12.5 = 87.5/12.5

The updated formula is then:

![Standard Error equals Design Factor times square root of the product of 87.5 over 12.5B and 50 squared]({{ site.baseurl }}/images/se_formula_modified.png)

I was able to calculate the median earnings estimate (and associated standard error and margin of error) within a few percent of the values given in the handbook. This provided me with confirmation that I was ready to expand my code to calculate median earnings estimates for other subgroups.

## The Stack

I built this app using the R package <a href="https://shiny.rstudio.com/reference/shiny/latest/">`Shiny`</a> which handles both the UI and the server. I store the data in a `sqlite` database and access it with queries written using the <a href="https://cran.r-project.org/web/packages/RSQLite/RSQLite.pdf">`RSQLite`</a> package which uses the <a href="https://dbi.r-dbi.org/reference/">DBI</a> API. The following section breaks down the R scripts based on functionality. Click on the script name to navigate to that section.

## The Codebase

- [`app.R`](#app-r)
  - UI and server functions to handle people inputs and plot/table/text outputs
- [`prep_db.R`](#prep-db-r)
  - Import, clean, combine and then load data into the `census_app_db.sqlite` database
- [`get_b20005_ruca_aggregate_earnings.R`](#get-b20005-ruca-aggregate-earnings-r)
  - Queries the database for earnings and associated margins of error for RUCA levels derived from Census Tracts
- [`calculate_median.R`](#calculate-median-r)
  - Derives estimate, standard of error and margin of error of median earnings for RUCA levels
- [`format_query_result.R`](#format-query-result-r)
  - Formats `calculate_median` query results
- [`get_b20005_tract_earnings.R`](#get-b20005-tract-earnings)
  - Queries the database for Census Tract-level earnings and associated margins of error 
- [`get_b20005_states.R`](#get-b20005-states-r)
  - Queries the SQLite database for a list of U.S. states 
- [`get_design_factor.R`](#get-design-factor-r)
  - Queries database for the design factor used for the median earnings estimation calculation
- [`get_b20005_labels.R`](#get-b20005-labels-r)
  - Queries the database for descriptive labels of B20005 table variables
- [`make_plot.R`](#make-plot-r)
  - Creates a bar plot object

## <a name="app-r"></a>`app.R`

A shiny app has three fundamental components:

```R
ui <- (...)
server <- (...)
shinyApp(ui, server,...)
```
The `ui` object holds all UI layout, input and output objects which define the front-end of your app. The `server` object holds all rendering functions which are assigned to outputs that appear on the UI. The `shinyApp` function takes a `ui` and `server` object (along with other arguments) and creates a shiny app object which can be run in a browser by passing it to the `runApp` function. Person inputs (such as selections in a dropdown) are assigned to a global `input` object.

### What's in my `ui`?

All of my UI objects are wrapped within a `fluidPage` call which returns a page layout which "consists of rows which in turn include columns" (from the [docs](https://shiny.rstudio.com/reference/shiny/latest/fluidPage.html)).

My app's UI has four sections:

1. Dropdowns to select state, sex and work status for which the person using the app wants ACS 5-year earnings estimates
2. A table with the estimate, standard error and margin of error for median earnings
3. A bar plot of population estimates for earnings levels for the selected state, sex, work status and RUCA (Rural-Urban Commuting Areas) level
4. A table with population estimates for earnings levels for each RUCA level for the selected state, sex and work status

Each section has a download button so that people can get the CSV files or plot image for their own analysis and reporting.
Each section is separated with `markdown('---')` which renders an HTML horizontal rule (`<hr>`).

#### Dropdowns

Dropdowns (the HTML `<select>` element) are a type of UI Input. I define each with an `inputId` which is a `character` object for reference on the server-side, a label `character` object which is rendered above the dropdown, and a `list` object which defines the dropdown options.

```
selectInput(
  inputId = "...",
  label = "...",
  choices = list(...)
)
```
In some cases, I want the person to see a `character` object in the dropdown that is more human-readable (e.g. `"Large Town"`) but use a corresponding input value in the server which is more computer-readable (e.g. `"Large_Town`). To achieve this, I use a named `character` vector where the names are displayed in the dropdown, and the assigned values are assigned to the global `input`:

```
selectInput(
     inputId = "ruca_level",
     label = "Select RUCA Level",
     choices = list(
       "RUCA LEVEL" = c(
       "Urban" = "Urban", 
       "Large Town" = "Large_Town", 
       "Small Town" = "Small_Town", 
       "Rural" = "Rural"))
     )
```
In this case, if the person selects `"Large Town"` the value assigned to `input$ruca_level` is `"Large_Town"`.


#### Tables

Tables (the HTML `<table>` element) are a type of UI Output. I define each with an `outputId` for reference in the server.

```
tableOutput(outputId = "...")
```

#### Plots

Similarly, a plot (which is rendered as an HTML `<img>` element) is a type of UI Output. I define each with an `outputId`.

```
plotOutput(outputId = "...")
```

#### Download Buttons
The download button (an HTML `<a>` element) is also a type of UI Output. I define each with an `outputId` and `label` (which is displayed as the HTML `textContent` attribute of the `<a>` element).

```
downloadButton(
  outputId = "...",
  label = "..."
)
```

### What's in my `server`?
The server function has three parameters: `input`, `output` and `session`. The `input` object is a `ReactiveValues` object which stores all UI Input values, which are accessed with `input$inputId`. The `output` object similarly holds UI Output values at `output$outputId`. I do not use the `session` object in my app (yet).

My app’s server has four sections:

1. Get data from the SQLite database
2. Render table and plot outputs
3. Prepare dynamic text (for filenames and the plot title)
4. Handle data.frame and plot downloads

#### Get data
There are three high-level functions which call query/format/calculation functions to return the data in the format necessary to produce table, text, download and plot outputs:

- The `earnings_data` function passes the person-selected dropdown options `input$sex`, `input$work_status` and `input$state` to the `get_b20005_ruca_aggregate_earnings` function to get a query result from the SQLite database. That function call is passed to `format_earnings`, which in turn is passed to the `reactive` function to make it a reactive expression. Only reactive expressions (and reactive endpoints in the `output` object) are allowed to access the `input` object which is a reactive source. You can read more about Shiny's "reactive programming model" in this [excellent article](https://shiny.rstudio.com/articles/reactivity-overview.html). 
```
earnings_data <- reactive(
  format_earnings(
    get_b20005_ruca_aggregate_earnings(
      input$sex, 
      input$work_status, 
      input$state)))
```

- The `design_factor` function passes the `input$state` selection to the `get_design_factor` function which in turn is passed to the `reactive` function.
```
design_factor <- reactive(get_design_factor(input$state))
```
- The `median_data` function passes the return values from `earnings_data()` and `design_factor()` to the `calculate_median` function which in turn is passed to the `reactive` function.
```
median_data <- reactive(calculate_median(earnings_data(), design_factor()))
```


#### Render Outputs
I have two reactive endpoints for table outputs, and one endpoint for a plot. The table outputs use `renderTable` (with row names displayed) with the `data.frame` coming from `median_data()` and `earnings_data()`. The plot output uses `renderPlot`, and a helper function `make_plot` to create a bar plot of `earnings_data()` for a person-selected `input$ruca_level` with a title created with the helper function `earnings_plot_title()`.
```
output$median_data <- renderTable(
  expr = median_data(), 
  rownames = TRUE)
  
output$earnings_data <- renderTable(
  expr = earnings_data(), 
  rownames = TRUE)
    
output$earnings_histogram <- renderPlot(
  expr = make_plot(
    data=earnings_data(), 
    ruca_level=input$ruca_level, 
    plot_title=earnings_plot_title()))
```

#### Prepare Dynamic Text
I created four functions that generate filenames for the `downloadHandler` call when the corresponding `downloadButton` gets clicked, one function that generates the title used to generate the bar plot, and one function which takes computer-readable `character` objects (e.g. `"Large_Town"`) and maps it to and returns a more human-readable `character` object (e.g. `"Large Town"`). I chose to keep filenames more computer-readable (to avoid spaces) and the plot title more human-readable.

```
get_pretty_text <- function(raw_text){
  text_map <- c("M" = "Male", 
  "F" = "Female",
  "FT" = "Full Time",
  "OTHER" = "Other",
  "Urban" = "Urban",
  "Large_Town" = "Large Town",
  "Small_Town" = "Small Town",
  "Rural" = "Rural")
  return(text_map[raw_text])
  }
 
earnings_plot_title <- function(){
  return(paste(
    input$state,
    get_pretty_text(input$sex),
    get_pretty_text(input$work_status),
    input$ruca_level,
    "Workers",
    sep=" "))
  }

b20005_filename <- function(){
    return(paste(
      input$state,
      get_pretty_text(input$sex),
      input$work_status,
      "earnings.csv",
      sep="_"
    ))
  }
  
median_summary_filename <- function() {
  paste(
    input$state,  
    get_pretty_text(input$sex), 
    input$work_status, 
    'estimated_median_earnings_summary.csv',  
    sep="_")
  }
  
ruca_earnings_filename <- function() {
  paste(
    input$state,  
    get_pretty_text(input$sex),  
    input$work_status, 
    'estimated_median_earnings_by_ruca_level.csv',  
    sep="_")
  }
  
earnings_plot_filename <- function(){
  return(paste(
    input$state,
    get_pretty_text(input$sex),
    input$work_status,
    input$ruca_level,
    "Workers.png",
    sep="_"))
  }
```

#### Handle downloads
I have five download buttons in my app: two which trigger a download of a zip file with two CSVs, two that downloads a single CSV, and one that downloads a single PNG. The `downloadHandler` function takes a `filename` and a `content` function to write data to a file.

In order to create a zip file, I use the `zip` base package function and pass it a vector with two filepaths (to which data is written using the base package's `write.csv` function) and a filename. I also specify the `contentType` as `"application/zip"`. In the zip file, one of the CSVs contains a query result from the `b20005` SQLite database table with earnings data, and the other file, `"b20005_variables.csv"` contains B20005 table variable names and descriptions. In order to avoid the files being written locally before download, I create a temporary directory with `tempdir` and prepend it to the filename to create the filepath.

For the bar plot image download, I use the `ggplot2` package's `ggsave` function, which takes a filename, a plot object (returned from the `make_plot` helper function) and the `character` object `"png"` (for the `device` parameter).

```
output$download_selected_b20005_data <- downloadHandler(
    filename = "b20005_data.zip",
    content = function(fname) {
      # Create a temporary directory to prevent local storage of new files
      temp_dir <- tempdir()
      
      # Create two filepath character objects and store them in a list
      # which will later on be passed to the `zip` function
      path1 <- paste(temp_dir, '/', b20005_filename(), sep="")
      path2 <- paste(temp_dir, "/b20005_variables.csv", sep="")
      fs <- c(path1, path2)
      
      # Create a CSV with person-selection input values and do not add a column
      # with row names
      write.csv(
        get_b20005_earnings(input$state, input$sex, input$work_status), 
        path1,
        row.names = FALSE)
      
      # Create a CSV for table B20005 variable names and labels for reference
      write.csv(
        get_b20005_ALL_labels(),
        path2,
        row.names = FALSE)
      
      # Zip together the files and add flags to maximize compression
      zip(zipfile = fname, files=fs, flags = "-r9Xj")
    },
    contentType = "application/zip"
  )
  
output$download_all_b20005_data <- downloadHandler(
  filename = "ALL_B20005_data.zip",
  content = function(fname){
    path1 <- "ALL_B20005_data.csv"
    path2 <- "b20005_variables.csv"
    fs <- c(path1, path2)
    
    write.csv(
      get_b20005_earnings('ALL', 'ALL', 'ALL'),
      path1,
      row.names = FALSE)
    
    write.csv(
      get_b20005_ALL_labels(),
      path2,
      row.names = FALSE)
    
    zip(zipfile = fname, files=fs, flags = "-r9Xj")
    },
    contentType = "application/zip"
  )
  
output$download_median_summary <- downloadHandler(
  filename = median_summary_filename(),
  content = function(file) {
    write.csv(median_data(), file)
    }
  )
  
output$download_earnings_plot <- downloadHandler(
  filename = earnings_plot_filename(),
  content = function(file) {
    ggsave(
      file, 
      plot = make_plot(
        data=earnings_data(), 
        ruca_level=input$ruca_level, 
        plot_title=earnings_plot_title()), 
        device = "png")
      }
  )
  
output$download_ruca_earnings <- downloadHandler(
  filename = ruca_earnings_filename(),
  content = function(file) {
    write.csv(earnings_data(), file)
  }
  )
```

### <a name="prep-db-r"></a>`prep_db.R`
The database diagram is shown below (created using <a href="https://dbdiagram.io">dbdiagram.io</a>):

![Database diag]({{ site.baseurl }}/images/census-app-db.jpg)


### <a name="get-b20005-ruca-aggregate-earnings-r"></a>`get_b20005_ruca_aggregate_earnings.R`

### <a name="calculate-median-r"></a>`calculate_median.R`

### <a name="format-query-result-r"></a>`format_query_result.R`

### <a name="get-b20005-tract-earnings-r"></a>`get_b20005_tract_earnings.R`

### <a name="get-b20005-states-r"></a>`get_b20005_earnings.R`

### <a name="get-design-factor-r"></a>`get_design_factor.R`

### <a name="get-b20005-labels-r"></a>`get_b20005_labels.R`

### <a name="make-plot-r"></a>`make_plot.R`

