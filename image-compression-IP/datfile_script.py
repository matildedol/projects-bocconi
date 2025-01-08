from PIL import Image
import sys
import itertools
import numpy

class darwin():

    def __init__(self, im, k):
        self.im = Image.open(im)
        self.k=k
        self.pixels = self.get_pixels()
        self.S=self.get_S()
        self.num_colors=self.get_number_colors()
        self.clusters=self.get_clusters()
        self.set_centers, self.dist_list=self.select_centers()
        
    def get_pixels(self):
        width, height = self.im.size
        pixels=[]
        for i in range(height):
            for j in range(width):
                x = self.im.getpixel((j,i))
                pixels.append(x)
        return pixels
    
    def get_S(self):
        S=[]
        for pixel in self.pixels:
            if pixel not in S:
                S.append(pixel)
        return S
    
    def get_number_colors(self):
        num_pixels=[]
        for s in self.S:
            num=self.pixels.count(s)
            num_pixels.append(num)
        return num_pixels
    
    def get_clusters(self):
        clusters=[]
        for r in range(1, len(self.S)+1):
            combinations=itertools.combinations(self.S,r)
            clusters.extend(combinations)
        return clusters

    def get_centroid(self, cluster, weights):
        centroid=[0,0,0]
        for coord in range(3):
            centroid[coord] = sum([(list(s)[coord]*weights[idx]) for idx,s in enumerate(self.S) if s in cluster])/sum([weights[idx] for idx,s in enumerate(self.S) if s in cluster])
        return tuple(centroid)

    def get_distance(self, c, cluster, weights):
        dist=0
        for idx,s in enumerate(self.S):
            if s in cluster:
                dist += (((c[0]-s[0])**2 + (c[1]-s[1])**2 + (c[2]-s[2])**2) * weights[idx])
        return dist

    def select_centers(self):
        survivors=[]
        distances=[]

        for cluster in self.clusters:
            if len(cluster)==0:
                continue
            elif len(cluster)==1:
                centroid=cluster[0]
            else:
                centroid=self.get_centroid(cluster, self.num_colors)

            dist=self.get_distance(centroid, cluster, self.num_colors)
            if dist <= 271295908.4427345:
                #print ('this centroid needs to be discarded:', centroid,'dissimilarity: ',dist )
                survivors.append(centroid)
                distances.append(dist)
                #print('loss is:', dist)            
        return survivors, distances
    
    def assignments_matrix(self):
        set_clusters=[cluster for cluster in self.clusters if self.get_centroid(cluster, self.num_colors) in self.set_centers]
        ass_matrix=numpy.zeros((len(self.S), len(self.set_centers)))
        for i,s in enumerate(self.S):
            for j,c in enumerate(set_clusters):
                if s in c:
                    ass_matrix[i,j]=1
        return ass_matrix



class writer():

    def __init__(self, set_colors, set_centers, k, dist_list, ass_matrix, dat_file_path):
        self.set_colors=set_colors
        self.set_centers=set_centers
        self.k=k
        self.dist_list=dist_list
        self.ass_matrix=ass_matrix
        self.dat_file_path=dat_file_path
    
    def write_dat_file(self):

        s_indices = list(range(1, len(self.set_colors) + 1))
        c_indices = list(range(1, len(self.set_centers) + 1))
        
        with open(self.dat_file_path, 'w') as f:
            f.write('data;\n\n')
            
            f.write('set COLORS :=\n')
            for i in s_indices:
                f.write(f'    {i}\n')
            f.write(';\n\n')
            f.write('set CENTERS :=\n')
            for j in c_indices:
                f.write(f'    {j}\n')
            f.write(';\n\n')

            f.write(f'param k := {self.k};\n\n')

            f.write('param dist :=\n')
            for j in c_indices:
                f.write(f'    {j} {self.dist_list[j-1]}\n') 
            f.write(';\n\n')
            
            f.write('param z :\n')
            f.write('    ')
            for j in c_indices:
                f.write(f'{j} ')
            f.write(':=\n')

            for i in s_indices:
                f.write(f'    {i} ')
                for j in c_indices:
                    f.write(f'{int(self.ass_matrix[i-1,j-1])} ')
                f.write('\n')
            f.write(';\n\n')

            
            f.write('end;\n')


def main():

    if len(sys.argv) != 4:
        print("Usage: python3 datfile_script.py {input-image} {k} {dat-file-path}")
        sys.exit(1)

    input_image_path = sys.argv[1]
    dat_file_path= sys.argv[3]

    try:
        k = int(sys.argv[3])
    except ValueError:
        print("Error: k must be an integer.")
        sys.exit(1)

    selected_data=darwin(input_image_path, k)

    print('\nthere are ',len(selected_data.set_centers),'possible centers')
    writer(selected_data.S,selected_data.set_centers, k, selected_data.dist_list, selected_data.assignments_matrix(), dat_file_path).write_dat_file()

if __name__ == "__main__":
    main()




