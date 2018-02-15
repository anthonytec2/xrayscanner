

#include <iostream>
#include <math.h>
#include <stdint.h>
#include <string>
using namespace std;

typedef struct Float3
{
    float x;
    float y;
    float z;
} float3;

float my_nextafterf(float a, int dir);
float onemove_in_cube(float3 *p0, float3 *v, float *htime, int *id);
void printFloat(float3 *pos, string name);
void readArr(float (*x)[208][256][225]);
int main()
{
    const uint8_t Nx = 208;                                                             //Number of voxels in x direction of target
    const uint16_t Ny = 256;                                                            //Number of voxels in y direction of target
    const uint8_t Nz = 225;                                                             //Number of voxels in z direction of target
    const uint8_t Mx = 200;                                                             //Number of pixels in x direction of detector
    const uint8_t My = 200;                                                             //Number of pixels in y direction of detector
    const uint8_t D = 2;                                                                //Size in mm of each pixel side
    const uint8_t h = 50;                                                               //height above detector of target
    const uint16_t H = h + Nz + 200;                                                    //height to reach detector
    const float3 orginOffset = {(-Mx * D) / 2 + (Nx / 2), (-My * D) / 2 + (Ny / 2), 0}; //offset for orgin of detector
    const float3 ep = {Nx / 2, Ny / 2, H};                                              //location of detector
    const float muBone = 0.573;                                                         //mu for bone in scan
    const float muFat = 0.193;                                                          //my for fat in scan
    float mu[Nx][Ny][Nz];                                                               //Mu array of voxels
    float detector[Mx][My];                                                             //Array of detector pixels
    float3 pos;                                                                         //position vector
    float3 dir;                                                                         //direction vector
    uint8_t i, j;                                                                       //loop counters for unrolled loop
    float L;                                                                            //energy count
    readArr(mu); 
    //cout<<mu[0][0][0];
    for (int ii = 0; ii < Mx * My; ii = ii + 1)
    {
        i = ii / Mx;
        j = ii - i * Mx;

        pos = {orginOffset.x + D * i, orginOffset.y + D * j, 0};
        dir = {ep.x - pos.x, ep.y - pos.y, ep.z - pos.z};
        L = sqrt(pow(dir.x, 2) + pow(dir.y, 2) + pow(dir.z, 2));
        dir.x = dir.x / L;
        dir.y = dir.y / L;
        dir.z = dir.z / L;
        L = 1;

        /*while (pos.z < h + Nz)
        {
            //onemove_in_cube(float3 * p0, float3 * v, float *htime, int *id)
            if (pos.x >= 0 && pos.x < Nx && pos.y >= 0 && pos.y < Ny && pos.z >= h && pos.z < h + Nz)
            {
                cout << "In the cube";
                L = L * exp(-1);
            }
        }
        detector[i][j] = L;
        */
    }
}

void readArr(float (*x)[256][225]){
    cout<<"Prefault I"<<endl;
    for(int i=0;i<208;i=i+1){
    for(int j=0;j<256;j=j+1){
	for(int z=0;z<225;z=z+1){
	x[i][j][z]=0;
	}
	}
	}
    //cout<<"x(0,0,0) "<<x[0][0][0]<<endl;
}

void printFloat(float3 *pos, string name)
{
    cout << name << ": " << pos->x << " " << pos->y << " " << pos->z << endl;
}

/**
Move photon to the first intersecting wall inside a 1x1x1mm cube
Input:
 p0: pointer to a float3 struct, {x,y,z}current x/y/z position of the ray
 v: pointer to a float3 struct, {vx,vy,vz}, direction unitary vector
Output:
 return value: the shortest distance to intersect with a voxel wall, in mm
 htime: pointer to a float[3] array, store the intersection position (x/y/z) of the first wall
 id: pointer to an integer, a flag: id=0: intersect with x-plane; id=1: y-plane; id=3: z-plane
*/
float onemove_in_cube(float3 *p0, float3 *v, float *htime, int *id)
{
    float dist, xi[3];
    //time-of-fly to hit the wall in each direction
    htime[0] = fabs((floor(p0->x) + (v->x > 0.f) - p0->x) / v->x);
    htime[1] = fabs((floor(p0->y) + (v->y > 0.f) - p0->y) / v->y);
    htime[2] = fabs((floor(p0->z) + (v->z > 0.f) - p0->z) / v->z);
    //get the direction with the smallest time-of-fly
    dist = fmin(fmin(htime[0], htime[1]), htime[2]);
    (*id) = (dist == htime[0] ? 0 : (dist == htime[1] ? 1 : 2));
    //p0 is inside, p is outside, move to the 1st intersection pt, now in the air side, to be corrected in the else block
    htime[0] = p0->x + dist * v->x;
    htime[1] = p0->y + dist * v->y;
    htime[2] = p0->z + dist * v->z;
    xi[0] = my_nextafterf((int)(htime[0] + 0.5f), (v->x > 0.f) - (v->x < 0.f)); // if using matlab, xi=round(htime)+eps(single(1.0))*(v>0 – v<0);
    xi[1] = my_nextafterf((int)(htime[1] + 0.5f), (v->y > 0.f) - (v->y < 0.f));
    xi[2] = my_nextafterf((int)(htime[2] + 0.5f), (v->z > 0.f) - (v->z < 0.f));
    if (*id == 0)
        htime[0] = xi[0];
    if (*id == 1)
        htime[1] = xi[1];
    if (*id == 2)
        htime[2] = xi[2];
    return dist;
}
float my_nextafterf(float a, int dir)
{
    union {
        float f;
        unsigned int i;
    } num;

    num.f = a + 1000.f;
    num.i += dir ^ (num.i & 0x80000000U);
    return num.f - 1000.f;
}
