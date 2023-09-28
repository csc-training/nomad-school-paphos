! Setup routines for heat equation solver
module setup
  use heat

contains

  subroutine initialize(previous, current, nsteps, parallel)
    use utilities

    implicit none

    type(field), intent(out) :: previous, current
    integer, intent(out) :: nsteps
    type(parallel_data), intent(out) :: parallel

    integer :: height, width, length
    logical :: using_input_file
    character(len=85) :: input_file, arg  ! Input file name and command line arguments


    ! Default values for grid size and time steps
    height = 800
    width = 800
    length = 800
    nsteps = 500
    using_input_file = .false.

    ! Read in the command line arguments and
    ! set up the needed variables
    select case(command_argument_count())
    case(0) ! No arguments -> default values
    case(4) ! Three arguments -> rows, cols and nsteps
       call get_command_argument(1, arg)
       read(arg, *) height
       call get_command_argument(2, arg)
       read(arg, *) width
       call get_command_argument(3, arg)
       read(arg, *) length
       call get_command_argument(4, arg)
       read(arg, *) nsteps
    case default
       call usage()
       stop
    end select

    ! Initialize the fields according the command line arguments
    call parallel_setup(parallel, height, width, length)
    call set_field_dimensions(previous, height, width, length, parallel)
    call set_field_dimensions(current, height, width, length, parallel)
    call generate_field(previous, parallel)
    call copy_fields(previous, current)

    if (parallel%rank == 0) then
       write(*,'(A, I5, A, I5, A, I5, A, I5)')  & 
            &  'Simulation parameters: height: ', height, ' width: ', width, &
            &  ' length: ', length, ' time steps: ', nsteps
       write(*,'(A, I5, A, I3, A, I3, A, I3, A)') 'Number of MPI tasks: ', & 
            &   parallel%size, ' (', parallel%dims(1), ' x ', parallel%dims(2), & 
            &   ' x ', parallel%dims(3), ' )'
       write(*,'(A, I5)') 'Number of threads: ', parallel%num_threads
    end if

  end subroutine initialize

  ! Generate initial the temperature field.  Pattern is disc with a radius
  ! of nx_full / 6 in the center of the grid.
  ! Boundary conditions are (different) constant temperatures outside the grid
  subroutine generate_field(field0, parallel)
    use heat

    implicit none

    type(field), intent(inout) :: field0
    type(parallel_data), intent(in) :: parallel

    real(dp) :: radius, di, dj, dk
    integer :: i, j, k, ds2

    ! The arrays for field contain also a halo region
    allocate(field0%data(0:field0%nx+1, 0:field0%ny+1, 0:field0%nz+1))

    ! Square of the disk radius
    radius = (field0%nx_full + field0%ny_full + field0%nz_full) / 18.0_dp

    !$omp parallel do private(i, j, k, di, dj, dk, ds2) collapse(2)
    do k = 0, field0%nz + 1
       do j = 0, field0%ny + 1
          do i = 0, field0%nx + 1
             di = i + parallel%coords(3) * field0%nx - field0%nx_full / 2.0_dp + 1
             dj = j + parallel%coords(2) * field0%ny - field0%ny_full / 2.0_dp + 1
             dk = k + parallel%coords(1) * field0%nz - field0%nz_full / 2.0_dp + 1
             ds2 = int(di**2 + dj**2 + dk**2)
             if (ds2 < radius**2) then
                field0%data(i,j,k) = 5.0_dp
             else
                field0%data(i,j,k) = 65.0_dp
             end if
          end do
       end do
    end do
    !$omp end parallel do

    ! Boundary conditions
    if (parallel%coords(1) == 0) then
       field0%data(:,:,0) = 20.0_dp
    end if
    if (parallel%coords(1) == parallel%dims(1) - 1) then
       field0%data(:,:,field0%nz+1) = 35.0_dp
    end if
    if (parallel%coords(2) == 0) then
       field0%data(:,0,:) = 20.0_dp
    end if
    if (parallel%coords(2) == parallel%dims(2) - 1) then
       field0%data(:,field0%ny+1,:) = 35.0_dp
    end if
    if (parallel%coords(3) == 0) then
       field0%data(0,:,:) = 20.0_dp
    end if
    if (parallel%coords(3) == parallel%dims(3) - 1) then
       field0%data(field0%nx+1,:,:) = 35.0_dp
    end if

  end subroutine generate_field


  ! Clean up routine for field type
  ! Arguments:
  !   field0 (type(field)): field variable to be cleared
  subroutine finalize(field0, field1)
    use heat

    implicit none

    type(field), intent(inout) :: field0, field1

    deallocate(field0%data)
    deallocate(field1%data)

  end subroutine finalize

  ! Helper routine that prints out a simple usage if
  ! user gives more than three arguments
  subroutine usage()
    implicit none
    character(len=256) :: buf

    call get_command_argument(0, buf)
    write (*,'(A)') 'Usage:'
    write (*,'(A, " (default values will be used)")') trim(buf)
    write (*,'(A, " <filename>")') trim(buf)
    write (*,'(A, " <filename> <nsteps>")') trim(buf)
    write (*,'(A, " <rows> <cols> <nsteps>")') trim(buf)
  end subroutine usage

end module setup
