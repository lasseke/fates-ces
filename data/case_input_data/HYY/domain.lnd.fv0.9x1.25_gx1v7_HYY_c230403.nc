CDF       
      nj        ni        nv              title         CESM domain data:      Conventions       CF-1.0     source        ;../gen_mapping_files/map_gx1v7_TO_fv0.9x1.25_aave.151008.nc    map_domain_a      4/glade/p/cesmdata/cseg/mapping/grids/gx1v7_151008.nc   map_domain_b      9/glade/p/cesmdata/cseg/mapping/grids/fv0.9x1.25_141008.nc      map_grid_file_ocn         4/glade/p/cesmdata/cseg/mapping/grids/gx1v7_151008.nc   map_grid_file_atm         9/glade/p/cesmdata/cseg/mapping/grids/fv0.9x1.25_141008.nc      
Created_on        
2023-04-03     
Created_by        lassetk    Created_with      ./subset_data -- 616905bbb     Created_from      T/cluster/shared/noresm/inputdata/share/domains/domain.lnd.fv0.9x1.25_gx1v7.151020.nc         xc               
_FillValue        �         	long_name         longitude of grid cell center      units         degrees_east   bounds        xv              
h   yc               
_FillValue        �         	long_name         latitude of grid cell center   units         degrees_north      bounds        yv     filter1       ) set_fv_pole_yc ON, yc = -+90 at j=1,j=nj               
p   xv                        
_FillValue        �         	long_name          longitude of grid cell verticies   coordinates       xc yc                
x   yv                        
_FillValue        �         	long_name         latitude of grid cell verticies    coordinates       xc yc                
�   mask                   	long_name         domain mask    note      unitless   comment       $0 value indicates cell is not active   coordinates       xc yc               
�   area                   
_FillValue        �         	long_name         $area of grid cell in radians squared   units         radian2    coordinates       xc yc               
�   frac                   
_FillValue        �         	long_name         $fraction of grid cell that is active   note      unitless   filter1       =error if frac> 1.0+eps or frac < 0.0-eps; eps = 0.1000000E-11      filter2       Jlimit frac to [fminval,fmaxval]; fminval= 0.1000000E-02 fmaxval=  1.000000     coordinates       xc yc               
�   ni                 
_FillValue        �                  
�   nj                  
_FillValue        �                  
�   lon                    
_FillValue        �         coordinates       xc yc               
�   lat                    
_FillValue        �         coordinates       xc yc               
�@7�     @N�&�}�L@7      @8`     @8`     @7      @N��sD[@N��sD[@OwI��z@OwI��z   ?&F�U& ?�      @7�     @N�&�}�L@7�     @N�&�}�L