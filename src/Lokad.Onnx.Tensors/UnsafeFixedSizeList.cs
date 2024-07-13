/*
 * Created by Galvanic Games (http://galvanicgames.com)
 * 
 * The MIT License (MIT)
 * 
 * Copyright (c) 2019
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *  
*/

using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

[StructLayout(LayoutKind.Sequential)]
public unsafe struct UnsafeFixedSizeList<T> : IList<T> where T : unmanaged
{
	public UnsafeFixedSizeList(T* ptr, int size)
	{
		this.ptr = ptr;
		this.size = size;

	}

	#region Implementation

	private T* ptr;
	private int size;
	private int _count;

	// Generated value by JetBrains
	private const int HASHCODE_MULTIPLIER = 397;
	private const int NO_INDEX = -1;

	public T this[int index]
	{
		get => *(ptr + index);
		set => *(ptr + index) = value;
	}

		
	public int Count => _count;
	public bool IsReadOnly => false;
	
	
	public IEnumerator<T> GetEnumerator()
	{
		for (int i = 0; i < _count; i++)
		{
			yield return this[i];
		}
	}

	IEnumerator IEnumerable.GetEnumerator()
	{
		return GetEnumerator();
	}
	
	public void Add(T item)
	{
		*(ptr + _count) = item;
		_count++;
	}

	public void AddRange(T[] arr)
	{		
		fixed (T* pThem = arr)
		{
			long copySizeInBytes = arr.Length * sizeof(T);
			
			Buffer.MemoryCopy(pThem, ptr + _count, copySizeInBytes, copySizeInBytes);
		}
		
		_count += arr.Length;
	}

	public void AddRange(List<T> list)
	{
		int listCount = list.Count;
			
		for (int i = 0; i < listCount; i++)
		{
			*(ptr + i + _count) = list[i];
		}		
	}

	public void Clear()
	{
		_count = 0;
	}

	public bool Contains(T item)
	{
		return IndexOf(item) != NO_INDEX;
	}

	public void CopyTo(T[] array, int arrayIndex)
	{
		fixed (T* pThem = array)
		{
			long sizeInBytes = _count * sizeof(T);
			
			Buffer.MemoryCopy(
				ptr, 
				pThem + arrayIndex,
				sizeInBytes,
				sizeInBytes);
		}
	}

	public bool Remove(T item)
	{
		int index = IndexOf(item);

		if (index != NO_INDEX)
		{
			RemoveAt(index);
			return true;
		}

		return false;
	}

	public int IndexOf(T item)
	{
			int size = sizeof(T);
			for (int i = 0; i < _count; i++)
			{
				// Similarly, if we later force T to implement IEquatable<T> then this should be replaced with
				// (pFirst + i)->Equals(item)
				if (MemoryCompare(ptr, &item, size))
				{
					return i;
				}
			}
		

		return NO_INDEX;
	}

	public void Insert(int index, T item)
	{
			_count++;
			
			for (int i = _count - 1; i >= index + 1; i--)
			{
				*(ptr + i) = *(ptr + i - 1);
			}

			*(ptr + index) = item;
		
	}

	public void RemoveAt(int index)
	{

		T* pItem = ptr + index;
		long copyAmountBytes = sizeof(T) * (_count - (index + 1));

		Buffer.MemoryCopy(
			pItem + 1,
			pItem,
			copyAmountBytes,
			copyAmountBytes);
		

		_count--;
	}

	public void UnstableRemoveAt(int index)
	{


			*(ptr + index) = *(ptr + _count - 1);
		
		
		_count--;
	}

	public bool Equals(UnsafeFixedSizeList<T> other)
	{
		if (_count != other._count)
		{
			return false;
		}
		else
		{


			// Below errors out with "You cannot use the fixed statement to take the address of an already fixed expression"
			// So I interpret that as I'm already safe? But I don't fully follow or understand so research is needed.
			// fixed(T* pFirstOther = &other._value0) {}
			return MemoryCompare(ptr, &other.ptr, _count * sizeof(T));
		}
	}

	public override int GetHashCode()
	{
		// If T doesn't implement IEquatable<T> this will generate garbage
		int hashCode = 0;

		unchecked
		{			
			for (int i = 0; i < _count; i++)
			{
				hashCode = (hashCode * HASHCODE_MULTIPLIER) ^ ((ptr + i)->GetHashCode());
			}
		}
		return hashCode;
	}

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool MemoryCompare(void* p1, void* p2, int sizeInBytes)
    {
        byte* pByte1 = (byte*)p1;
        byte* pByte2 = (byte*)p2;

        // There are some interesting solutions here, possibly faster ones?
        // https://stackoverflow.com/questions/43289/comparing-two-byte-arrays-in-net
        for (int i = 0; i < sizeInBytes; i++)
        {
            if (*pByte1 != *pByte2)
            {
                return false;
            }

            pByte1++;
            pByte2++;
        }

        return true;
    }
    #endregion
}
