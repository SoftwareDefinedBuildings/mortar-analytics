# Get available points in the Brick schema of a site

This app obtains and saves all the points and the point types in the Brick schema of a particular site (or all sites).
This could be useful for users to know what type of data they have access to in a particular building, which helps them in developing analytics. 
The output from this app could also be used to evaluate the completeness of the Brick schema of a particular site. 

## Run this app
`python app.py [-site <site>]`

* `site`: if specified, get points for this particular site (default: None [get points from all sites])

## Output

This produces a CSV file having the following syntax: `points_{site}.csv` (default: `points_all.csv`) when run. Each row is contains the following information:
* point: name of the point in the site
* type: brick type of that point
* site: site to which the point belongs to