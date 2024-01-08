extern "C" {
    void testingfor(double* x);   // C++
}
//extern void testingfor_(double* x);  // C 

int main(int argc, char* argv[]) {
    double y = 1.0;
 
    testingfor(&y);
}