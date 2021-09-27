---
layout: post
description: A tutorial to create a geodatabase, maps and layouts to visualize U.S. Census Data in ArcGIS Pro
categories: [markdown]
title: Visualize U.S. Census Data with ArcGIS
---

By Vishal Bakshi

In this blog post, I’ll walk through my process of creating an ArcGIS geodatabase and a set of layouts visualizing U.S. Census Data. The data used for this app is from table B20005 (Sex By Work Experience In The Past 12 Months By Earnings In The Past 12 Months).

## Table of Contents

- [Get the Data](#get-the-data)
  - [Tract Boundaries](#tract-boundaries-1)
  - [ACS 5-Year Estimates](#acs-5-year-estimates-1)
  - [Using data.census.gov](#using-data-census-gov)
  - [Using the censusapi R package](#using-censusapi)
- [Connect Data to Geodatabase](#connect-data-to-geodatabase)
  - [Tract Boundaries](#tract-boundaries-2)
  - [ACS 5-Year Estimates](#acs-5-year-estimates-2)
- [Visualize Data](#visualize-data)
  - [Create a Map](#create-a-map)
  - [Create a Symbology](#create-a-symbology)
  - [Create a Layout](#create-a-layout)
- [Normalize the Data](#normalize-the-data)
  - [Create Additional Layouts](#create-additional-layouts)

## Get the Data<a name=""></a>

### Tract Boundaries<a name=""></a>

- Download and unzip 2019 TIGER Shapefile for MN (tl_2019_27_tract.zip) (corresponds to the final year, 2019, in the ACS 5-year estimates). These will contain the Census Tract geographies needed to create a map in ArcGIS.

### ACS 5-Year Estimates<a name=""></a>

#### Using data.census.gov<a name=""></a>

- On data.census.gov, search for B20005

<img src="{{ site.baseurl }}/images/arcgis_01.png" width="50%" />

- Select the link to the Table B20005 with “2019 inflation-adjusted dollars”

<img src="{{ site.baseurl }}/images/arcgis_02.png" width="50%" />

- Click the dropdown at the top next to the label **Product** and select _2015: ACS 5-Year Estimates Detailed Tables_

<img src="{{ site.baseurl }}/images/arcgis_03.png" width="50%" />

- Click **Customize Table** at the top right of the page

<img src="{{ site.baseurl }}/images/arcgis_04.png" width="50%" />

- In the **Geo*** section, click _Tract > Minnesota > All Census Tracts within Minnesota_

<img src="{{ site.baseurl }}/images/arcgis_05.png" width="50%" />

- Once it’s finished loading, click **Close** and then **Download Table**

<img src="{{ site.baseurl }}/images/arcgis_06.png" width="50%" />

- Once downloaded, extract the zip folder and open the file _ACSDT52015.B20005_data_with_overlays_….xslx_ in Excel any tool that can handle tabular data

- Slice the last 11 characters of the _GEO_ID_ (using the **RIGHT** function in a new column) to replace the existing _GEO_ID_ column values. For example, a GEO_ID of _1400000US27029000100_ should be replaced with _27029000100_. This will later on be matched with the _GEOID_ field in the _tl_2019_27_tract_ shapefile

- Save/export the file as .XLSX

#### Using the `censusapi` R package<a name=""></a>

Pass the following arguments to the `censusapi::listCensusMetadata` function and assign its return value to `B20005_vars`:

<br>

```R
B20005_vars <- censusapi::listCensusMetadata(
  name="acs/acs5",
  vintage="2015",
  type="variables",
  group="B20005"
)
```
<br>

- Pass the following arguments to censusapi::getCensus and assign its return value to B20005:
- 
<br>

```R
B20005 <- censusapi::listCensusMetadata(
  name="acs/acs5",
  vintage="2015",
  region="tract:*",
  regionin="state:27", # 27 = Minnesota state FIPS code
  vars=c("GEO_ID", "NAME", B20005_vars$name)
)
```
<br>

- Replace _GEO_ID_ (or create a new column) with the last 11 characters

<br>

```R
B20005 <- substr(B20005$GEO_ID, 10, 20)
```

<br>

- Export to an .XLSX file

<br>

```R
write.xlsx(B20005, “acs5_b20005_minnesota.xlsx”, row.names = FALSE)
```

<br>

## Connect Data to Geodatabase<a name=""></a>

Open ArcGIS Pro and start a new project.

### Tract Boundaries<a name=""></a>

- Right click _Folders_ in the **Contents** pane and click _Add folder_ connection

<img src="{{ site.baseurl }}/images/arcgis_07.png" width="50%" />

- Select the downloaded (and extracted) _tl_2019_27_tract_ folder and click **OK**

<img src="{{ site.baseurl }}/images/arcgis_08.png" width="50%" />

- Click on _tl_2019_27_tract_ folder in the **Contents** pane

- In the **Catalog** pane, right-click _tl_2019_27.shp_ and then click _Export > Feature Class to Geodatabase_

<img src="{{ site.baseurl }}/images/arcgis_09.png" width="50%" />

- Confirm _Input Features_ (tl_2019_27_tract.shp) and _Output Geodatabase_ (Default.gdb or whatever geodatabase you are connected to) and then click the green **Run** button

- Refresh the Geodatabase and click on it in the **Contents** pane to view the added shapefile

<img src="{{ site.baseurl }}/images/arcgis_10.png" width="50%" />

### ACS 5-Year Estimates<a name=""></a>

- Under the **View** ribbon click on _Geoprocessing_ to open that pane

- In the **Geoprocessing** pane, search for _Join Field_ and click on it

<img src="{{ site.baseurl }}/images/arcgis_11.png" width="50%" />


<img src="{{ site.baseurl }}/images/arcgis_12.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_13.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_14.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_15.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_16.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_17.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_18.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_19.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_20.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_21.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_22.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_23.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_24.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_25.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_26.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_27.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_28.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_29.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_30.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_31.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_32.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_33.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_34.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_35.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_36.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_37.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_38.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_39.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_40.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_41.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_42.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_43.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_44.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_45.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_46.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_47.png" width="50%" />

