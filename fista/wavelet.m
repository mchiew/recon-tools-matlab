classdef wavelet < matlab.mixin.Copyable
    properties
        name
        scale
        transp
        S
    end
    properties ( Access = private )
    end
    methods
        function b = wavelet(name, scale)
            b.name = name;
            b.scale = scale;
            b.transp = 0;
            b.S = [];
        end
        function b = mtimes(obj, a)
            if obj.transp
                br = waverec2(real(a), obj.S, obj.name); 
                bi = waverec2(imag(a), obj.S, obj.name); 
                b  = br + 1j*bi;
            else
                [br, obj.S] = wavedec2(real(a), obj.scale, obj.name);        
                [bi, obj.S] = wavedec2(imag(a), obj.scale, obj.name);        
                b = br + 1j*bi;
            end
        end
        function b = ctranspose(obj)
            b   =   copy(obj);
            b.transp = xor(obj.transp, 1); 
        end
    end
end
