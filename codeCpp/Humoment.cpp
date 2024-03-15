#include <iostream>
#include <math.h>
using namespace std;


int main()
{
    int binnaryImage[5][6] = {  0, 1, 0, 0, 0, 0,
                                0, 1, 1, 0, 0, 0,
                                0, 1, 1, 1, 0, 0,
                                0, 1, 1, 1, 0, 0,
                                0, 0, 0, 0, 0, 0};

    //print image
    for (int y = 0; y < 5; y++)
    {
        for (int x = 0; x < 6 ; x++)
            cout << binnaryImage[y][x];
        cout << endl;
    }
        
    // m00
    int m00 = 0;
    float centroidx = 0, centroidy = 0;
    for (int y = 0; y < 5; y++)
        for (int x = 0; x < 6 ; x++)
        {
            m00 += binnaryImage[y][x];
            if (binnaryImage[y][x] == 1) {
                centroidy += (y + 1); 
                centroidx += (x + 1);
            }
        }    
    centroidx /= float(m00);
    centroidy /= float(m00);
    cout << "m00 = " << m00 << endl <<"centroidx = " << centroidx << endl << "centroidy = " << centroidy << endl;
    
    float M20, M02, M11, M30, M03, M12, M21;
    float mqp1 =0, mqp2 = 0, mqp3 = 0, mqp4 =0, mqp5 = 0, mqp6 = 0, mqp7 = 0;
    
    int q0 = 0, q1 = 1, q2 = 2, q3 = 3;
    int p0 = 0, p1 = 1, p2 = 2, p3 = 3;
     
    for (int x = 0; x < 5; x++)
    {    for (int y = 0; y < 6; y++)
        {
            if (binnaryImage[x][y] == 1) 
            {   
                // M20
                mqp1 += powf((float(x + 1) - centroidx), p2)*powf((float(y + 1) - centroidy), q0); 
                // M02
                mqp2 += powf((float(x + 1) - centroidx), p0)*powf((float(y + 1) - centroidy), q2); 
                // M11
                mqp3 += powf((float(x + 1) - centroidx), p1)*powf((float(y + 1) - centroidy), q1); 
                // M30
                mqp4 += powf((float(x + 1) - centroidx), p3)*powf((float(y + 1) - centroidy), q0); 
                // M12
                mqp5 += powf((float(x + 1) - centroidx), p1)*powf((float(y + 1) - centroidy), q2); 
                // M03
                mqp6 += powf((float(x + 1) - centroidx), p0)*powf((float(y + 1) - centroidy), q3);
                // M21 
                mqp7 += powf((float(x + 1) - centroidx), p2)*powf((float(y + 1) - centroidy), q1); 
            }
        
        }
    }
    
    M20 = float(mqp1/(powf(m00,(p2 + q0)/2 +1)));        
    M02 = float(mqp2/(powf(m00,(p0 + q2)/2 +1)));
    M11 = float(mqp2/(powf(m00,(p1 + q1)/2 +1)));
    M30 = float(mqp2/(powf(m00,(p3 + q0)/2 +1)));
    M03 = float(mqp2/(powf(m00,(p0 + q3)/2 +1)));
    M12 = float(mqp2/(powf(m00,(p1 + q2)/2 +1)));
    M21 = float(mqp2/(powf(m00,(p2 + q1)/2 +1)));
    
    float S1 = float(M20 + M02);
    float S2 = float((M20 + M02)*(M20 - M02)+ 4*M11*M11);
    float S3 = float(powf(M30 - 3*M12,2) + powf(M30 - 3*M21,2));
    float S4 = float(powf(M30 + M12, 2) + powf(M03 + M21,2));
    float S5 = float((M30 - 3*M12)*(M30 + M12)*(powf(M30 + M12,2) - 3*powf(M03 + M21,2) 
                + (3*M21 - M03)*(M03 + M21)*(3*powf(M30 + M12,2) - powf(M03 + M21,2))));
    float S6 = float((M20 - M02)*(powf(M30 + M12,2) - powf(M03 + M21, 2)) + 4*M11*(M30 + M12)*(M03 + M21));
    float S7 = float((3*M21 - M03)*(M30 + M12)*(powf(M30 + M12,2) - 3*powf(M03 + M21,2)) + (M30 - 3*M12)*(M21 + M02)*(3*powf(M30 +M12,2) - powf(M03 + M12,2)));
    
    cout << "S1 = " << S1 << endl;
    cout << "S2 = " << S2 << endl;
    cout << "S3 = " << S3 << endl;
    cout << "S4 = " << S4 << endl;
    cout << "S5 = " << S5 << endl;
    cout << "S6 = " << S6 << endl;    
    cout << "S7 = " << S7 << endl;

    return 0;
}