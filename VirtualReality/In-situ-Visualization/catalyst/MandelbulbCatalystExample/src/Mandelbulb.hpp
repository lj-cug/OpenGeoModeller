#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <mpi.h>
#ifdef USE_HDF5
#include <hdf5.h>
#endif

class Mandelbulb {

    struct Vector {
        double x;
        double y;
        double z;
    };

    inline int& value_at(unsigned x, unsigned y, unsigned z) {
        return m_data[z + m_depth*(y + m_height*x)];
    }

    static inline double do_r(const Vector& v)
    {
        return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    }

    static inline double do_theta(double n, const Vector& v)
    {
        return n * std::atan2(v.y,v.x);
    }

    static inline double do_phi(double n, const Vector& v)
    {
        return n * std::asin(v.z/do_r(v));
    }

    static int iterate(const Vector& v0, double order, unsigned max_cycles)
    {
        Vector v = v0;
        double n = order;
        int i;
        for(i=0; i < max_cycles && (v.x*v.x+v.y*v.y+v.z*v.z < 2.0); i++) {
            double r         = do_r(v);
            double theta     = do_theta(n,v);
            double phi       = do_phi(n,v);
            double rn        = std::pow(r,n);
            double cos_theta = std::cos(theta);
            double sin_theta = std::sin(theta);
            double cos_phi   = std::cos(phi);
            double sin_phi   = std::sin(phi);
            Vector vn = {
                rn * cos_theta * cos_phi,
                rn * sin_theta * cos_phi,
                -rn * sin_phi
            };
            v.x = vn.x + v0.x;
            v.y = vn.y + v0.y;
            v.z = vn.z + v0.z;
        }
        return i;
    }

    public:

    Mandelbulb(
            unsigned width,
            unsigned height,
            unsigned depth,
            double z_offset,
            float range=1.2,
            unsigned nprocs=1)
    : m_width(width)
    , m_height(height)
    , m_depth(depth+1)
    , m_extents(6)
    , m_origin(3)
    , m_data(width*height*(depth+1))
    , m_z_offset(z_offset)
    , m_range(range)
    , m_nprocs(nprocs) {
        m_extents[4] = 0;
        m_extents[5] = m_width-1;
        m_extents[2] = 0;
        m_extents[3] = m_height-1;
        m_extents[0] = 0;
        m_extents[1] = m_depth-1;
        m_origin[2] = 0;
        m_origin[1] = 0;
        m_origin[0] = z_offset/nprocs;
    }

    Mandelbulb(const Mandelbulb&)            = default;
    Mandelbulb(Mandelbulb&&)                 = default;
    Mandelbulb& operator=(const Mandelbulb&) = default;
    Mandelbulb& operator=(Mandelbulb&&)      = default;
    ~Mandelbulb()                            = default;

    void compute(double order, unsigned max_cycles=100) {
        for(unsigned z = 0; z < m_depth;  z++)
        for(unsigned y = 0; y < m_height; y++)
        for(unsigned x = 0; x < m_width;  x++)
        {
            Vector v = {
                2.0*m_range*x/m_width - m_range,
                2.0*m_range*y/m_height - m_range,
                2.0*m_range*(m_z_offset+z)/((m_depth-1)*m_nprocs) - m_range
            };
            value_at(x,y,z) = iterate(v, order, max_cycles);
        }
    }

    void writeBIN(const std::string& filename) {
        std::ofstream ofile(filename.c_str(), std::ios::binary);
        ofile.write((char*)m_data.data(), sizeof(int)*m_data.size());
    }

    int* GetExtents() const {
        return const_cast<int*>(m_extents.data());
    }

    double* GetOrigin() const {
        return const_cast<double*>(m_origin.data());
    }

    int* GetData() const {
        return const_cast<int*>(m_data.data());
    }

    int GetNumberOfLocalCells() const {
        return m_data.size();
    }

    private:

    size_t              m_width;
    size_t              m_height;
    size_t              m_depth;
    std::vector<int>    m_extents;
    std::vector<double> m_origin;
    std::vector<int>    m_data;
    const unsigned      m_z_offset;
    const float         m_range;
    const unsigned      m_nprocs;
};

