from PIL import Image
import random
import sys

class kmeans_image():
    def __init__(self,im,k):
        self.im = Image.open(im)
        self.k=k
        self.pixels=self.get_pixels()
        self.S=self.get_S()
        self.num=self.get_number_colors()
    
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
    
    def random_centers(self):
        C = list(tuple(random.randint(0, 255) for _ in range(3)) for _ in range(self.k))
        return C
    
    def get_number_colors(self):
        num_pixels=[]
        for s in self.S:
            num=self.pixels.count(s)
            num_pixels.append(num)
        return num_pixels
    
    def assignments(self, C):
        assignments={}
        for idx,x in enumerate(self.S):
            delta = [float('inf'), 0]
            for c in C:
                distance = ((c[0]-x[0])**2 + (c[1]-x[1])**2 + (c[2]-x[2])**2 )*self.num[idx]
                if distance < delta[0]:
                    delta[0] = distance
                    delta[1] = c
            assignments[x] = delta[1]
        return assignments
    
    def centroids(self, C):
        assignments=self.assignments(C)
        new_C=[]
        for c in C:
            cluster=[x for x in self.pixels if assignments[x]==c]
            centroid=(0,0,0)
            if len(cluster)>0:
                centroid_temp=list(centroid)
                for coord in range(3):
                    centroid_temp[coord]= sum([x[coord] for x in cluster])/len(cluster) #non round
                centroid=tuple(centroid_temp)
                new_C.append(centroid)
            else:
                new_C.append(centroid)
        return new_C
    
    def cluster_is_diff(self,C,new_C):
        if C!=new_C:
            return True
        return False

    def lloyd(self):
        C=self.random_centers()
        self.assignments(C)
        new_C=self.centroids(C)
        while self.cluster_is_diff(C,new_C):
            C=new_C.copy()
            self.assignments(C)
            new_C=self.centroids(C)
        final_assignments=self.assignments(new_C)
        return final_assignments, new_C
    

class recolor_image():

    def __init__(self, input_image, output_file, new_colours):
        self.input_image=Image.open(input_image)
        self.output_file=output_file
        self.new_colours=new_colours

    def new_image(self, pixels):
        new_pixels = [tuple(map(round, self.new_colours[pixel])) for pixel in pixels]
        new_im = Image.new('RGB', self.input_image.size)
        new_im.putdata(new_pixels)
        return new_im.save(self.output_file)
    
class tot_dissimilarity():

    def __init__(self, final_values, pixels, final_assignments_dict):
        self.final_centers=final_values
        self.pixels=pixels
        self.final_assignments_dict=final_assignments_dict

    def get_dissim(self):
        dissim=0
        for x in self.pixels:
            for c in self.final_centers:
                if self.final_assignments_dict[x]==c:
                    dissim += (c[0]-x[0])**2 + (c[1]-x[1])**2 + (c[2]-x[2])**2
        return dissim


def main():

    if len(sys.argv) != 4:
        print("Usage: python3 recolor.py {input-image} {output-image} {k}")
        sys.exit(1)

    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]

    try:
        k = int(sys.argv[3])
    except ValueError:
        print("Error: k must be an integer.")
        sys.exit(1)
    
    kmeans=kmeans_image(input_image_path, k)
    pixels=kmeans.pixels
    final_assignments, final_centers=kmeans.lloyd()
    new_im=recolor_image(input_image_path, output_image_path, final_assignments)
    new_im.new_image(pixels)
    final_dissim=tot_dissimilarity(final_centers, pixels, final_assignments)
    print('The final dissimilarity is ', final_dissim.get_dissim())

    
if __name__ == "__main__":
    main()

