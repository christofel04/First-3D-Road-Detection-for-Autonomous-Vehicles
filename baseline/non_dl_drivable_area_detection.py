import numpy as np
import math

def gaussian_kernel_weight( distance : float , treshold_radius_grid : int = 4 ) :

    treshold_radius = treshold_radius_grid * math.sqrt(2)

    if distance < treshold_radius :
    
        return 0
    
    else :

        return (((2 + math.cos(2* math.pi*distance/treshold_radius))/3) + (1 - ( distance / treshold_radius )) + ( math.sin( 2*math.pi*distance/treshold_radius )/(2*math.pi)))



def drivable_area_prediction_using_non_dl_lidar( lidar_pts , voxel_size = [ 0.2 , 0.2 ] , lidar_range = [-50 , 50 , -50 , 70 ] , HEIGHT_MEAN_TRESHOLD = -1 , HEIGHT_VARIANCE_TRESHOLD = 0.01 , DIFFERENT_MAX_MIN_HEIGHT_TRESHOLD = 0.2 , BAYESIAN_GAUSSIAN_KERNEL_RADIUS = 4 ) :

    if lidar_range[0] < 0 :
        lidar_range[1] = lidar_range[1] + -1* lidar_range[0]
        lidar_pts[ : , 0 ] = lidar_pts[ : , 0 ] + -1 * lidar_range[0]
        lidar_range[0] = 0        

        

    if lidar_range[2] < 0 :
        lidar_range[3] = lidar_range[3] + -1* lidar_range[2]
        lidar_pts[ : , 1 ] = lidar_pts[ : , 1 ] + -1* lidar_range[2]
        lidar_range[2] = 0       

        

    
    length_of_pillar_bev = int( ( lidar_range[ 1] - lidar_range[0] )/ voxel_size[0] )
    width_of_pillar_bev = int( ( lidar_range[ 3 ] - lidar_range[2] )/ voxel_size[1] )
    
    list_of_points_in_tensor = [[[] for i in range( width_of_pillar_bev ) ] for j in range( length_of_pillar_bev )]

    for lidar_point in lidar_pts :

        X_coordinate_lidar_point = int( lidar_point[0]//voxel_size[0] )

        if (( X_coordinate_lidar_point < 0 ) | ( X_coordinate_lidar_point >= length_of_pillar_bev ))  :
            continue
            
        Y_coordinate_lidar_point = int( lidar_point[1]//voxel_size[1] )

        if (( Y_coordinate_lidar_point < 0 ) | ( Y_coordinate_lidar_point >= width_of_pillar_bev )) :
            continue

        #print( "Select lidar points : " + str( lidar_point ))

        list_of_points_in_tensor[ X_coordinate_lidar_point][ Y_coordinate_lidar_point ].append( lidar_point )

    # Measure mean and height variance of every grid
        
    list_of_points_mean_and_variance_height = [[[None , None , None ] for i in range( width_of_pillar_bev ) ] for j in range( length_of_pillar_bev )]
        
    for x,y in range( length_of_pillar_bev ), range( width_of_pillar_bev ) :

        if len( list_of_points_in_tensor[x][y] ) > 0 :

            list_of_points_mean_and_variance_height[x][y][0] = np.mean(np.array( list_of_points_in_tensor[x][y][2] ))

            list_of_points_mean_and_variance_height[x][y][1] = np.std(np.array( list_of_points_in_tensor[x][y][2] ))

            list_of_points_mean_and_variance_height[x][y][2] = np.max(np.array( list_of_points_in_tensor[x][y][2] )) - np.min(np.array( list_of_points_in_tensor[x][y][2] ))

    # Classify drivable area and non- drivable area using mean height and height variance of object in the grid
            
    list_of_ground_and_none_ground_mean_and_variance_height = [[None for i in range( width_of_pillar_bev ) ] for j in range( length_of_pillar_bev )]
            
    for x,y in range( length_of_pillar_bev ), range( width_of_pillar_bev ) :

        if (( list_of_points_mean_and_variance_height[x][y][0] is not None ) & ( list_of_points_mean_and_variance_height[x][y][1] is not None )) :

            if list_of_points_mean_and_variance_height[x][y][0] > HEIGHT_MEAN_TRESHOLD :

                list_of_ground_and_none_ground_mean_and_variance_height[x][y] = 1 

                continue

            elif list_of_points_mean_and_variance_height[x][y][1] >= HEIGHT_VARIANCE_TRESHOLD :

                list_of_ground_and_none_ground_mean_and_variance_height[x][y] = 0

            elif list_of_ground_and_none_ground_mean_and_variance_height[x][y][2] >= DIFFERENT_MAX_MIN_HEIGHT_TRESHOLD :

                list_of_ground_and_none_ground_mean_and_variance_height[x][y] = 0

    # Predict drivable area or non drivable area of grid using Bayesian Gaussian Kernel
                
    refined_list_of_point_mean_and_variance_height = list_of_points_mean_and_variance_height.copy()
                
    for x,y in range( length_of_pillar_bev ), range( width_of_pillar_bev ) :

        if list_of_ground_and_none_ground_mean_and_variance_height[x][y] is None :

            sum_of_weighted_mean_and_variance_height = 0

            sum_of_gaussian_kernel_weight_of_mean_and_variance_surround_grid = 0

            for x_gaussian_kernel, y_gaussian_kernel in range( x- BAYESIAN_GAUSSIAN_KERNEL_RADIUS , x + BAYESIAN_GAUSSIAN_KERNEL_RADIUS + 1 ), range( y - BAYESIAN_GAUSSIAN_KERNEL_RADIUS , y + BAYESIAN_GAUSSIAN_KERNEL_RADIUS + 1 ):

                if ((x_gaussian_kernel < 0 ) or ( x_gaussian_kernel >= length_of_pillar_bev ) or ( y_gaussian_kernel < 0 ) or ( y_gaussian_kernel >= width_of_pillar_bev )) :

                    continue
                    
                if ((list_of_points_mean_and_variance_height[x_gaussian_kernel][y_gaussian_kernel][0] is None ) or (list_of_points_mean_and_variance_height[ x_gaussian_kernel ][ y_gaussian_kernel ][1] is None )) :

                    continue

                sum_of_weighted_mean_and_variance_height = sum_of_weighted_mean_and_variance_height + list_of_points_mean_and_variance_height[x_gaussian_kernel][y_gaussian_kernel ] * gaussian_kernel_weight( math.sqrt( (x_gaussian_kernel - x)**2 + ( y_gaussian_kernel - y )**2) , BAYESIAN_GAUSSIAN_KERNEL_RADIUS )

                sum_of_gaussian_kernel_weight_of_mean_and_variance_surround_grid = sum_of_gaussian_kernel_weight_of_mean_and_variance_surround_grid +  gaussian_kernel_weight( math.sqrt( (x_gaussian_kernel - x)**2 + ( y_gaussian_kernel - y )**2) , BAYESIAN_GAUSSIAN_KERNEL_RADIUS )

            
            if sum_of_weighted_mean_and_variance_height == 0 :

                # Then grid doesnt have any drivable area around the grid

                continue

            else :

                refined_list_of_point_mean_and_variance_height[x][y][0] = sum_of_weighted_mean_and_variance_height/sum_of_gaussian_kernel_weight_of_mean_and_variance_surround_grid

    
    for x,y in range( length_of_pillar_bev ), range( width_of_pillar_bev ) :

        if list_of_ground_and_none_ground_mean_and_variance_height[x][y] is None :

            if refined_list_of_point_mean_and_variance_height[x][y][0] is None :

                list_of_ground_and_none_ground_mean_and_variance_height[x][y] = 0

            else :

                if refined_list_of_point_mean_and_variance_height[x][y] >= HEIGHT_MEAN_TRESHOLD :

                    list_of_ground_and_none_ground_mean_and_variance_height[x][y] = 0

                else :

                    list_of_ground_and_none_ground_mean_and_variance_height[x][y] = 1 


    assert None not in list_of_ground_and_none_ground_mean_and_variance_height


    return np.array( list_of_ground_and_none_ground_mean_and_variance_height )






                                                           
                                                        



                
    






