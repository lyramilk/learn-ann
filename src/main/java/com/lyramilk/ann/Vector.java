package com.lyramilk.ann;

public class Vector implements java.io.Serializable {
    public double[] data;

    public Vector(int size) {
        data = new double[size];
    }

    public Vector(double[] data) {
        this.data = data;
    }


    public Vector addAndAssign(Vector v) {
        if (data.length != v.data.length) {
            throw new RuntimeException("size not match");
        }
        for (int i = 0; i < data.length; i++) {
            data[i] += v.data[i];
        }
        return this;
    }

    public Vector addAndAssign(double v) {
        for (int i = 0; i < data.length; i++) {
            data[i] += v;
        }
        return this;
    }

    public Vector sub(Vector v) {
        if (data.length != v.data.length) {
            throw new RuntimeException("size not match");
        }
        for (int i = 0; i < data.length; i++) {
            data[i] -= v.data[i];
        }
        return this;
    }

    public Vector mul(double v) {
        for (int i = 0; i < data.length; i++) {
            data[i] *= v;
        }
        return this;
    }

    public Vector hadamard(Vector v) {
        if (data.length != v.data.length) {
            throw new RuntimeException("size not match");
        }
        for (int i = 0; i < data.length; i++) {
            data[i] *= v.data[i];
        }
        return this;
    }

    public Vector div(double v) {
        for (int i = 0; i < data.length; i++) {
            data[i] /= v;
        }
        return this;
    }

    public Vector pow(double v) {
        for (int i = 0; i < data.length; i++) {
            data[i] = Math.pow(data[i], v);
        }
        return this;
    }

    public double sum() {
        double sum = 0;
        for (int i = 0; i < data.length; i++) {
            sum += data[i];
        }
        return sum;
    }

    public double dot(Vector v) {
        if (data.length != v.data.length) {
            throw new RuntimeException("size not match");
        }
        double sum = 0;
        for (int i = 0; i < data.length; i++) {
            sum += data[i] * v.data[i];
        }
        return sum;
    }

    public double norm() {
        double sum = 0;
        for (int i = 0; i < data.length; i++) {
            sum += data[i] * data[i];
        }
        return Math.sqrt(sum);
    }

    public Vector normalize() {
        double n = norm();
        if (n == 0) {
            return this;
        }
        return div(n);
    }

    public Vector copy() {
        double[] newData = new double[data.length];
        System.arraycopy(data, 0, newData, 0, data.length);
        return new Vector(newData);
    }

    public Vector assign(Vector v) {
        if (data.length != v.data.length) {
            throw new RuntimeException("size not match");
        }
        System.arraycopy(v.data, 0, data, 0, data.length);
        return this;
    }

    public int size() {
        return data.length;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < data.length; i++) {
            sb.append(data[i]);
            if (i < data.length - 1) {
                sb.append(",");
            }
        }
        sb.append("]");
        return sb.toString();
    }
}
