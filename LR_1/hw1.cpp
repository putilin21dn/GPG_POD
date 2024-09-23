#include <iostream>
#include <algorithm>

using namespace std;


int main() {

    float a,b,c;
    cin >> a >> b >> c;

    if (a==0 && b == 0 && c == 0)
    {
        cout << "any" << endl;     
    }
    else{
        if (a==0 && b == 0){
            cout << "incorrect" << endl; 
        }
        else{
            if (a == 0){
                cout << -c / b << endl;
            }
        }
    }


    cout.precision(6);

    float D = pow(b, 2) -4*a*c;

    if (D > 0){

        cout << (-b + pow(D, 0.5)) / (2 * a) << " " << (-b - pow(D, 0.5)) / (2 * a) << endl;
        
    }
    else{
        if (D == 0){

            cout << -b / (2 * a) << endl;
        
        }
        else{
            if (D < 0){
            cout << "imaginary" << endl;
            
            }
        }
    }
}