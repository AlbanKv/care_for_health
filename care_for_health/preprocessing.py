from haversine import haversine, Unit
import pandas as pd

### ---------------- Neighbors calculations: ---------------------------
# Calculate closest neighbors with calculate_cluster_row, 
# Perform the operation on the entire dataset with calculate_cluster_df,
# Apply computations (med requirements + available) on each code_insee

def calculate_cluster_row(row, df, radius=30):
    '''
    Return a list of neirest neighbors to a given INSEE code
    By default, radius=30 km
    '''
    neighbors=[]
    for i in range(len(df)):
        if haversine((row.Lat_commune, row.Lon_commune), (df.loc[i,'Lat_commune'], df.loc[i,'Lon_commune'])) <= radius:
            neighbors.append(df.loc[i,'code_insee'])
    row.neighbors = neighbors
    return row

def calculate_cluster_df(df, radius=30):
    '''
    Return the DataFrame with reseted index (!), containing the list of neirest neighbors
    By default, radius=30 km
    '''
    df_ = df.copy().reset_index(drop=True)
    df_['neighbors']=0
    df_ = df_.apply(lambda row: calculate_cluster_row(row, df_, radius=radius), axis=1)
    return df_

def get_meds_neighbors(df):
    '''
    Compute insee and medical informations, based on the nearest neighbors.
    Returns a DataFrame.
    '''
    df_return = df.copy()
    df_merge = pd.DataFrame(columns=["code_insee", "neighbors_Besoin_medecins", "neighbors_nb_medecins", "neighbors_diff_meds"])
    for row in df_return.iterrows():
        sum_besoin_meds = df_return.loc[(df_return["code_insee"].isin(row[1]["neighbors"]))]["Besoin_medecins"].sum()
        sum_meds = df_return.loc[(df_return["code_insee"].isin(row[1]["neighbors"]))]["Medecin_generaliste"].sum()
        diff_meds = sum_besoin_meds - sum_meds
        df_merge = df_merge.append({"code_insee": row[1]["code_insee"],
                                    "neighbors_Besoin_medecins": sum_besoin_meds,
                                   "neighbors_nb_medecins": sum_meds,
                                   "neighbors_diff_meds": diff_meds},
                                   ignore_index=True)
    return df_return.merge(df_merge, on="code_insee")

### ---------------- IsInPolygon Calculations: -------------------------
# Transforms STR polygon to a list of 2-tuples
# Answers the question 'Is the point in polygon?' with 'is_inside_polygon'

def polygon_to_list(str_polygon):
    str_polygon = str_polygon[10:-2]
    poly_liste = str_polygon.split(", ")
    new_list = []
    for coord in poly_liste:
        new_list.append(tuple(map(float, coord.split(' '))))
    return new_list
 
# Given three collinear points p, q, r, the function checks if point q lies on line segment 'pr'
def onSegment(p:tuple, q:tuple, r:tuple) -> bool:
     
    if ((q[0] <= max(p[0], r[0])) &
        (q[0] >= min(p[0], r[0])) &
        (q[1] <= max(p[1], r[1])) &
        (q[1] >= min(p[1], r[1]))):
        return True
         
    return False
 
# To find orientation of ordered triplet (p, q, r). 
#  The function returns following values 
# 0 --> p, q and r are collinear
# 1 --> Clockwise
# 2 --> Counterclockwise
def orientation(p:tuple, q:tuple, r:tuple) -> int:
    val = (((q[1] - p[1]) *
            (r[0] - q[0])) -
           ((q[0] - p[0]) *
            (r[1] - q[1])))
    if val == 0:
        return 0
    if val > 0:
        return 1 # Collinear
    else:
        return 2 # Clock or counterclock

def doIntersect(p1, q1, p2, q2):
    # Find the four orientations needed for 
    # general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    # General case
    if (o1 != o2) and (o3 != o4):
        return True
    # Special Cases
    # p1, q1 and p2 are collinear and
    # p2 lies on segment p1q1
    if (o1 == 0) and (onSegment(p1, p2, q1)):
        return True
    # p1, q1 and p2 are collinear and
    # q2 lies on segment p1q1
    if (o2 == 0) and (onSegment(p1, q2, q1)):
        return True
    # p2, q2 and p1 are collinear and
    # p1 lies on segment p2q2
    if (o3 == 0) and (onSegment(p2, p1, q2)):
        return True
    # p2, q2 and q1 are collinear and
    # q1 lies on segment p2q2
    if (o4 == 0) and (onSegment(p2, q1, q2)):
        return True
    return False

# is Medecin in Code_insee?
def is_inside_polygon(points:list, p:tuple) -> bool:
    n = len(points)
    # There must be at least 3 vertices
    # in polygon
    if n < 3:
        return False
    # Create a point for line segment
    # from p to infinite
    extreme = (10_000, p[1]) #INT_MAX currently set at 10_000, can be higher
    count = i = 0
    while True:
        next = (i + 1) % n
        # Check if the line segment from 'p' to 
        # 'extreme' intersects with the line 
        # segment from 'polygon[i]' to 'polygon[next]'
        if (doIntersect(points[i],
                        points[next],
                        p, extreme)):
            # If the point 'p' is collinear with line 
            # segment 'i-next', then check if it lies 
            # on segment. If it lies, return true, otherwise false
            if orientation(points[i], p,
                           points[next]) == 0:
                return onSegment(points[i], p,
                                 points[next])
            count += 1
        i = next
        if (i == 0):
            break
    # Return true if count is odd, false otherwise
    return (count % 2 == 1)
